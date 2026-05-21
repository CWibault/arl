from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from ogb_impls.utils.encoders import GCEncoder
from ogb_impls.utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from ogb_impls.utils.networks import (
    MLP,
    GCActor,
    GCValue,
    Identity,
    LengthNormalize,
    length_normalize,
)


class HIQL1vAgent(flax.struct.PyTreeNode):
    """Hierarchical implicit Q-learning agent with one value function (HIQL1v)."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def value_loss(self, batch, grad_params):
        """Compute the value loss."""

        (next_v1_t, next_v2_t) = self.network.select("target_value")(
            batch["next_observations"], batch["high_value_goals"]
        )
        next_v_t = jnp.minimum(next_v1_t, next_v2_t)
        q = (
            batch["high_value_rewards"]
            + self.config["discount"] * batch["high_value_masks"] * next_v_t
        )

        (v1_t, v2_t) = self.network.select("target_value")(
            batch["observations"], batch["high_value_goals"]
        )
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = (
            batch["high_value_rewards"]
            + self.config["discount"] * batch["high_value_masks"] * next_v1_t
        )
        q2 = (
            batch["high_value_rewards"]
            + self.config["discount"] * batch["high_value_masks"] * next_v2_t
        )
        (v1, v2) = self.network.select("value")(
            batch["observations"], batch["high_value_goals"], params=grad_params
        )
        v = (v1 + v2) / 2

        value_loss1 = self.expectile_loss(adv, q1 - v1, self.config["expectile"]).mean()
        value_loss2 = self.expectile_loss(adv, q2 - v2, self.config["expectile"]).mean()
        value_loss = value_loss1 + value_loss2

        return value_loss, {
            "value_loss": value_loss,
            "v_mean": v.mean(),
            "v_max": v.max(),
            "v_min": v.min(),
        }

    def high_critic_loss(self, batch, grad_params):
        """Compute the high-level critic loss."""
        (next_v1, next_v2) = self.network.select("value")(
            batch["high_actor_actions"], batch["high_actor_goals"]
        )
        next_v = (next_v1 + next_v2) / 2
        q = next_v

        if self.config["use_hiql_rep"]:
            high_actor_actions = self.network.select("goal_rep")(
                jnp.concatenate(
                    [batch["observations"], batch["high_actor_actions"]], axis=-1
                )
            )
        else:
            high_actor_actions = batch["high_actor_actions"]

        q1, q2 = self.network.select("high_critic")(
            batch["observations"],
            batch["high_actor_goals"],
            high_actor_actions,
            params=grad_params,
        )
        critic_loss = ((q1 - q) ** 2 + (q2 - q) ** 2).mean()

        return critic_loss, {
            "critic_loss": critic_loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
        }

    def low_critic_loss(self, batch, grad_params):
        """Compute the low-level critic loss."""
        next_v1, next_v2 = self.network.select("value")(
            batch["next_observations"], batch["low_value_goals"]
        )
        next_v = (next_v1 + next_v2) / 2
        q = (
            batch["low_value_rewards"]
            + self.config["discount"] * batch["low_value_masks"] * next_v
        )

        q1, q2 = self.network.select("low_critic")(
            batch["observations"],
            batch["low_value_goals"],
            batch["actions"],
            params=grad_params,
        )
        critic_loss = ((q1 - q) ** 2 + (q2 - q) ** 2).mean()

        return critic_loss, {
            "critic_loss": critic_loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
        }

    def low_actor_loss(self, batch, grad_params, rng=None):
        """Compute the low-level actor loss."""

        if self.config["use_hiql_rep"]:
            # --- compute the goal representations of the subgoals ---
            goal_reps = self.network.select("goal_rep")(
                jnp.concatenate(
                    [batch["observations"], batch["low_actor_goals"]], axis=-1
                ),
                params=grad_params,
            )
            if not self.config["low_actor_grad"]:
                # --- stop gradients through the goal representations ---
                goal_reps = jax.lax.stop_gradient(goal_reps)
        else:
            goal_reps = batch["low_actor_goals"]
        dist = self.network.select("low_actor")(
            batch["observations"], goal_reps, goal_encoded=True, params=grad_params
        )
        log_prob = dist.log_prob(batch["actions"])

        if self.config["low_actor_loss"] == "awr":
            v1, v2 = self.network.select("value")(
                batch["observations"], batch["low_actor_goals"]
            )
            nv1, nv2 = self.network.select("value")(
                batch["next_observations"], batch["low_actor_goals"]
            )
            v = (v1 + v2) / 2
            nv = (nv1 + nv2) / 2
            adv = nv - v

            exp_a = jnp.exp(adv * self.config["low_awr_alpha"])
            exp_a = jnp.minimum(exp_a, self.config["low_awr_clip"])
            actor_loss = -(exp_a * log_prob).mean()

            actor_info = {
                "actor_loss": actor_loss,
                "adv": adv.mean(),
                "bc_log_prob": log_prob.mean(),
                "mse": jnp.mean((dist.mode() - batch["actions"]) ** 2),
                "std": jnp.mean(dist.scale_diag),
            }
            return actor_loss, actor_info

        elif self.config["low_actor_loss"] == "ddpgbc":
            # --- ddpgbc loss. ---
            if self.config["actor_const_std"]:
                sampled_actions = jnp.clip(dist.mode(), -1, 1)
            else:
                sampled_actions = jnp.clip(dist.sample(seed=rng), -1, 1)
            q1, q2 = self.network.select("low_critic")(
                batch["observations"],
                batch["low_actor_goals"],
                sampled_actions,
            )
            q = jnp.minimum(q1, q2)

            # --- normalise q values by absolute mean to make loss scale invariant. ---
            q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)

            bc_loss = -(self.config["low_ddpgbc_alpha"] * log_prob).mean()

            actor_loss = q_loss + bc_loss

            return actor_loss, {
                "actor_loss": actor_loss,
                "q_loss": q_loss,
                "bc_loss": bc_loss,
                "q_mean": q.mean(),
                "q_abs_mean": jnp.abs(q).mean(),
                "bc_log_prob": log_prob.mean(),
                "mse": jnp.mean((dist.mode() - batch["actions"]) ** 2),
                "std": jnp.mean(dist.scale_diag),
            }
        else:
            raise ValueError(
                f"Unsupported low-level actor loss: {self.config["low_actor_loss"]}"
            )

    def high_actor_loss(self, batch, grad_params, rng=None):
        """Compute the high-level actor loss."""

        dist = self.network.select("high_actor")(
            batch["observations"], batch["high_actor_goals"], params=grad_params
        )
        if self.config["use_hiql_rep"]:
            high_actor_actions = self.network.select("goal_rep")(
                jnp.concatenate(
                    [batch["observations"], batch["high_actor_actions"]], axis=-1
                )
            )
        else:
            high_actor_actions = batch["high_actor_actions"]
        log_prob = dist.log_prob(high_actor_actions)

        if self.config["high_actor_loss"] == "awr":
            v1, v2 = self.network.select("value")(
                batch["observations"], batch["high_actor_goals"]
            )
            nv1, nv2 = self.network.select("value")(
                batch["high_actor_next_observations"], batch["high_actor_goals"]
            )
            v = (v1 + v2) / 2
            nv = (nv1 + nv2) / 2
            adv = nv - v

            exp_a = jnp.exp(adv * self.config["high_awr_alpha"])
            exp_a = jnp.minimum(exp_a, self.config["high_awr_clip"])
            actor_loss = -(exp_a * log_prob).mean()

            return actor_loss, {
                "actor_loss": actor_loss,
                "adv": adv.mean(),
                "bc_log_prob": log_prob.mean(),
                "mse": jnp.mean((dist.mode() - high_actor_actions) ** 2),
                "std": jnp.mean(dist.scale_diag),
            }

        elif self.config["high_actor_loss"] == "ddpgbc":
            # --- ddpgbc loss. ---
            if self.config["actor_const_std"]:
                sampled_actions = dist.mode()
            else:
                sampled_actions = dist.sample(seed=rng)

            # --- normalise high_actor_actions to unit length. ---
            if self.config["use_hiql_rep"]:
                sampled_actions = length_normalize(sampled_actions)
            q1, q2 = self.network.select("high_critic")(
                batch["observations"],
                batch["high_actor_goals"],
                sampled_actions,
            )
            q = jnp.minimum(q1, q2)

            # --- normalise q values by absolute mean to make loss scale invariant. ---
            q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)

            bc_loss = -(self.config["high_ddpgbc_alpha"] * log_prob).mean()

            actor_loss = q_loss + bc_loss

            return actor_loss, {
                "actor_loss": actor_loss,
                "q_loss": q_loss,
                "bc_loss": bc_loss,
                "q_mean": q.mean(),
                "q_abs_mean": jnp.abs(q).mean(),
                "bc_log_prob": log_prob.mean(),
                "mse": jnp.mean((dist.mode() - high_actor_actions) ** 2),
                "std": jnp.mean(dist.scale_diag),
            }
        else:
            raise ValueError(
                f"Unsupported high-level actor loss: {self.config["high_actor_loss"]}"
            )

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, high_actor_rng, low_actor_rng = jax.random.split(rng, 3)

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f"value/{k}"] = v

        high_critic_loss, high_critic_info = self.high_critic_loss(batch, grad_params)
        for k, v in high_critic_info.items():
            info[f"high_critic/{k}"] = v

        low_critic_loss, low_critic_info = self.low_critic_loss(batch, grad_params)
        for k, v in low_critic_info.items():
            info[f"low_critic/{k}"] = v

        low_actor_loss, low_actor_info = self.low_actor_loss(
            batch, grad_params, low_actor_rng
        )
        for k, v in low_actor_info.items():
            info[f"low_actor/{k}"] = v

        high_actor_loss, high_actor_info = self.high_actor_loss(
            batch, grad_params, high_actor_rng
        )
        for k, v in high_actor_info.items():
            info[f"high_actor/{k}"] = v

        loss = (
            value_loss
            + high_critic_loss
            + low_critic_loss
            + low_actor_loss
            + high_actor_loss
        )
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1 - self.config["tau"]),
            self.network.params[f"modules_{module_name}"],
            self.network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, "value")

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor."""
        high_seed, low_seed = jax.random.split(seed)

        high_dist = self.network.select("high_actor")(
            observations, goals, temperature=temperature
        )
        goal_reps = high_dist.sample(seed=high_seed)
        if self.config["use_hiql_rep"]:
            goal_reps = (
                goal_reps
                / jnp.linalg.norm(goal_reps, axis=-1, keepdims=True)
                * jnp.sqrt(goal_reps.shape[-1])
            )

        low_dist = self.network.select("low_actor")(
            observations, goal_reps, goal_encoded=True, temperature=temperature
        )
        actions = low_dist.sample(seed=low_seed)

        actions = jnp.clip(actions, -1, 1)

        return actions

    @classmethod
    def create(
        cls,
        seed,
        example_batch,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            example_batch: Example batch.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_observations = example_batch["observations"]
        ex_actions = example_batch["actions"]
        ex_goals = example_batch["high_actor_goals"]
        action_dim = ex_actions.shape[-1]
        goal_dim = ex_goals.shape[-1]
        ex_options = jnp.zeros(shape=(1, config["rep_dim"]))

        if config["use_hiql_rep"]:
            # --- define (state-dependent) subgoal representation phi([s; g]) that outputs a length-normalized vector ---
            goal_rep_seq = []
            goal_rep_seq.append(
                MLP(
                    hidden_dims=(*config["rep_hidden_dims"], config["rep_dim"]),
                    activate_final=False,
                    layer_norm=config["layer_norm"],
                )
            )
            goal_rep_seq.append(LengthNormalize())
            goal_rep_def = nn.Sequential(goal_rep_seq)

            value_encoder_def = GCEncoder(
                state_encoder=Identity(), concat_encoder=goal_rep_def
            )
            target_value_encoder_def = GCEncoder(
                state_encoder=Identity(), concat_encoder=goal_rep_def
            )
            # --- low-level actor: pi^l(. | s, phi([s; w])) ---
            low_actor_encoder_def = GCEncoder(
                state_encoder=Identity(), concat_encoder=goal_rep_def
            )
            # --- high-level actor: pi^h(. | s, g) (i.e., no encoder) ---
            high_actor_encoder_def = None
        else:
            value_encoder_def = None
            target_value_encoder_def = None
            low_actor_encoder_def = None
            high_actor_encoder_def = None

        value_def = GCValue(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            num_ensembles=2,
            gc_encoder=value_encoder_def,
        )
        target_value_def = GCValue(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            num_ensembles=2,
            gc_encoder=target_value_encoder_def,
        )
        high_critic_def = GCValue(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            num_ensembles=2,
        )
        low_critic_def = GCValue(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            num_ensembles=2,
        )
        high_actor_def = GCActor(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=config["rep_dim"] if config["use_hiql_rep"] else goal_dim,
            state_dependent_std=False,
            const_std=config["actor_const_std"],
            gc_encoder=high_actor_encoder_def,
        )
        low_actor_def = GCActor(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=action_dim,
            state_dependent_std=False,
            const_std=config["actor_const_std"],
            gc_encoder=low_actor_encoder_def,
        )

        network_info = dict(
            value=(value_def, (ex_observations, ex_goals)),
            target_value=(target_value_def, (ex_observations, ex_goals)),
            high_critic=(high_critic_def, (ex_observations, ex_goals, ex_options)),
            low_critic=(low_critic_def, (ex_observations, ex_goals, ex_actions)),
            low_actor=(low_actor_def, (ex_observations, ex_goals)),
            high_actor=(high_actor_def, (ex_observations, ex_goals)),
        )
        if config["use_hiql_rep"]:
            network_info.update(
                goal_rep=(
                    goal_rep_def,
                    (jnp.concatenate([ex_observations, ex_goals], axis=-1)),
                ),
            )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params["modules_target_value"] = params["modules_value"]

        config["low_discount"] = 1 - 1 / config["subgoal_steps"]
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # --- agent hyperparameters. ---
            agent_name="hiql1v",  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            layer_norm=True,  # Whether to use layer normalization.
            # --- representation hyperparameters. ---
            rep_dim=10,  # Goal representation dimension.
            use_hiql_rep=True,  # Whether to use goal representation.
            rep_hidden_dims=(
                512,
                512,
                512,
            ),  # Representation network hidden dimensions.
            # --- value function hyperparameters. ---
            value_hidden_dims=(
                1024,
                1024,
                1024,
                1024,
            ),  # Value network hidden dimensions.
            discount=0.995,  # Discount factor.
            tau=0.005,  # Target network update rate.
            expectile=0.7,  # IQL expectile.
            # --- actor hyperparameters. ---
            actor_hidden_dims=(
                1024,
                1024,
                1024,
                1024,
            ),  # Actor network hidden dimensions.
            actor_const_std=True,  # Whether to use constant standard deviation for the actors.
            low_actor_grad=False,  # Whether to use gradient for the low-level actor.
            low_actor_loss="awr",  # Low-level actor loss type ("awr" or "ddpgbc").
            low_awr_alpha=3.0,  # Low-level AWR temperature.
            low_ddpgbc_alpha=0.1,  # Low-level DDPG+BC temperature.
            low_awr_clip=100.0,  # Low-level AWR clip.
            high_actor_loss="awr",  # High-level actor loss type ("awr" or "ddpgbc").
            high_awr_alpha=3.0,  # High-level AWR temperature.
            high_ddpgbc_alpha=0.1,  # High-level DDPG+BC temperature.
            high_awr_clip=100.0,  # High-level AWR clip.
            # --- dataset hyperparameters. ---
            dataset_class="HGCDataset",  # Dataset class name.
            subgoal_steps=25,  # Actor subgoal steps.
            value_p_curgoal=0.2,  # Probability of using the current state as the value goal.
            value_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.3,  # Probability of using a random state as the value goal.
            value_geom_sample=False,  # Whether to use geometric sampling for future value goals.
            # --- [only required for HGCDataset] ---
            low_value_p_curgoal=0.10,  # Probability of using the current state as the value goal.
            low_value_p_trajgoal=0.85,  # Probability of using a future state in the same trajectory as the value goal.
            low_value_p_randomgoal=0.05,  # Probability of using a random state as the value goal.
            low_value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            # --- [only required for HGCDataset] ---
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.5,  # Probability of using a random state as the actor goal.
            actor_geom_sample=True,  # Whether to use geometric sampling for future actor goals.
            gc_negative=True,  # Whether to use "0 if s == g else -1" (True) or "1 if s == g else 0" (False) as reward.
        )
    )
    return config
