import copy
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from ogb_impls.utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from ogb_impls.utils.networks import (
    GCActor,
    GCValue,
    HGCEncoder,
    soft_length_normalize,
    length_normalize,
)


class ARLAgent(flax.struct.PyTreeNode):
    """Abstractive Reinforcement Learning (ARL) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def high_value_loss(self, batch, grad_params):
        """Compute the high-level value loss."""
        (next_v1_t, next_v2_t) = self.network.select("target_high_value")(
            batch["next_observations"], batch["high_value_goals"]
        )
        next_v_t = jnp.minimum(next_v1_t, next_v2_t)
        q = (
            batch["high_value_rewards"]
            + self.config["discount"] * batch["high_value_masks"] * next_v_t
        )

        (v1_t, v2_t) = self.network.select("target_high_value")(
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
        (v1, v2) = self.network.select("high_value")(
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
        (next_v1, next_v2) = self.network.select("high_value")(
            batch["high_actor_actions"], batch["high_actor_goals"]
        )
        next_v = (next_v1 + next_v2) / 2
        q = next_v

        if self.config["option_diff"]:
            high_actor_options = self.network.select("option_rep")(
                goals=batch["high_actor_actions"] - batch["observations"],
            )
        else:
            high_actor_options = self.network.select("option_rep")(
                goals=jnp.concatenate([batch["observations"], batch["high_actor_actions"]], axis=-1),
            )

        q1, q2 = self.network.select("high_critic")(
            batch["observations"],
            batch["high_actor_goals"],
            high_actor_options,
            params=grad_params,
        )
        critic_loss = ((q1 - q) ** 2 + (q2 - q) ** 2).mean()

        return critic_loss, {
            "critic_loss": critic_loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
        }

    def low_value_loss(self, batch, grad_params):
        """Compute the low-level value loss."""
        if self.config["critic_diff"]:
            q1, q2 = self.network.select("target_low_critic")(
                batch["observations"],
                batch["low_value_goals"] - batch["observations"],
                actions=batch["actions"],
            )
        else:
            q1, q2 = self.network.select("target_low_critic")(
                batch["observations"],
                batch["low_value_goals"],
                actions=batch["actions"],
            )
        q = jnp.minimum(q1, q2)

        if self.config["value_diff"]:
            v = self.network.select("low_value")(
                batch["low_value_goals"] - batch["observations"], params=grad_params
            )
        else:
            v = self.network.select("low_value")(
                batch["observations"], batch["low_value_goals"], params=grad_params
            )

        value_loss = self.expectile_loss(q - v, q - v, self.config["expectile"]).mean()

        return value_loss, {
            "value_loss": value_loss,
            "v_mean": v.mean(),
            "v_max": v.max(),
            "v_min": v.min(),
        }

    def low_critic_loss(self, batch, grad_params):
        """Compute the low-level critic loss."""
        if self.config["value_diff"]:
            next_v = self.network.select("low_value")(
                batch["low_value_goals"] - batch["next_observations"]
            )
        else:
            next_v = self.network.select("low_value")(
                batch["next_observations"],
                batch["low_value_goals"],
            )
        q = (
            batch["low_value_rewards"]
            + self.config["low_discount"] * batch["low_value_masks"] * next_v
        )

        if self.config["critic_diff"]:
            q1, q2 = self.network.select("low_critic")(
                batch["observations"],
                batch["low_value_goals"] - batch["observations"],
                actions=batch["actions"],
                params=grad_params,
            )
        else:
            q1, q2 = self.network.select("low_critic")(
                batch["observations"],
                batch["low_value_goals"],
                actions=batch["actions"],
                params=grad_params,
            )
        critic_loss = ((q1 - q) ** 2 + (q2 - q) ** 2).mean()

        return critic_loss, {
            "critic_loss": critic_loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
        }

    def high_actor_loss(self, batch, grad_params, rng=None):
        """Compute the high-level actor loss."""
        
        # --- compute the high-level goal representations. ---
        dist = self.network.select("high_actor")(
            batch["observations"], batch["high_actor_goals"], params=grad_params
        )
        if self.config["option_diff"]:
            options = self.network.select("option_rep")(
                goals=batch["high_actor_actions"] - batch["observations"]
            )
        else:
            options = self.network.select("option_rep")(
                goals=jnp.concatenate([batch["observations"], batch["high_actor_actions"]], axis=-1),
            )
        log_prob = dist.log_prob(options)

        if self.config["high_actor_loss"] == "awr":
            q1, q2 = self.network.select("high_critic")(
                batch["observations"],
                batch["high_actor_goals"],
                options,
            )
            v1, v2 = self.network.select("high_value")(
                batch["observations"],
                batch["high_actor_goals"],
            )
            v = (v1 + v2) / 2
            q = jnp.minimum(q1, q2)
            adv = q - v

            exp_a = jnp.exp(adv * self.config["high_awr_alpha"])
            exp_a = jnp.minimum(exp_a, self.config["high_awr_clip"])

            actor_loss = -(exp_a * log_prob).mean()

            return actor_loss, {
                "actor_loss": actor_loss,
                "adv": adv.mean(),
                "bc_log_prob": log_prob.mean(),
                "mse": jnp.mean((dist.mode() - options) ** 2),
                "std": jnp.mean(dist.scale_diag),
            }

        elif self.config["high_actor_loss"] == "ddpgbc":
            # --- ddpgbc loss. ---
            if self.config["actor_const_std"]:
                sampled_actions = dist.mode()
            else:
                sampled_actions = dist.sample(seed=rng)

            # --- normalise options to unit length. ---
            if self.config["option_diff"]:
                sampled_actions = soft_length_normalize(sampled_actions)
            else:
                sampled_actions = length_normalize(sampled_actions)
            q1, q2 = self.network.select("high_critic")(
                batch["observations"], batch["high_actor_goals"], sampled_actions
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
                "mse": jnp.mean((dist.mode() - options) ** 2),
                "std": jnp.mean(dist.scale_diag),
            }

    def low_actor_loss(self, batch, grad_params, rng=None):
        """Compute the low-level actor loss."""
        
        # --- compute the low-level goal representations. ---
        if self.config["option_diff"]:
            options = self.network.select("option_rep")(
                goals=batch["low_actor_goals"] - batch["observations"], params=grad_params
            )
        else:
            options = self.network.select("option_rep")(
                goals=jnp.concatenate([batch["observations"], batch["low_actor_goals"]], axis=-1), params=grad_params
            )

        dist = self.network.select("low_actor")(
            batch["observations"], options, params=grad_params
        )
        log_prob = dist.log_prob(batch["actions"])

        # --- low_critic / low_value expect raw goals (same dim as at init), not option_rep outputs. ---

        if self.config["low_actor_loss"] == "awr":
            # --- awr loss. ---
            if self.config["critic_diff"]:
                q1, q2 = self.network.select("low_critic")(
                    batch["observations"],
                    batch["low_actor_goals"] - batch["observations"],
                    actions=batch["actions"],
                )
            else:
                q1, q2 = self.network.select("low_critic")(
                    batch["observations"],
                    batch["low_actor_goals"],
                    actions=batch["actions"],
                )
            if self.config["value_diff"]:
                v = self.network.select("low_value")(
                    batch["low_actor_goals"] - batch["observations"]
                )
            else:
                v = self.network.select("low_value")(
                    batch["observations"],
                    batch["low_actor_goals"],
                )
            q = jnp.minimum(q1, q2)
            adv = q - v

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
            if self.config["critic_diff"]:
                q1, q2 = self.network.select("low_critic")(
                    batch["observations"],
                    batch["low_actor_goals"] - batch["observations"],
                    actions=sampled_actions,
                )
            else:
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

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, high_actor_rng, low_actor_rng = jax.random.split(rng, 3)

        high_value_loss, high_value_info = self.high_value_loss(batch, grad_params)
        for k, v in high_value_info.items():
            info[f"high_value/{k}"] = v

        high_critic_loss, high_critic_info = self.high_critic_loss(batch, grad_params)
        for k, v in high_critic_info.items():
            info[f"high_critic/{k}"] = v

        low_value_loss, low_value_info = self.low_value_loss(batch, grad_params)
        for k, v in low_value_info.items():
            info[f"low_value/{k}"] = v

        low_critic_loss, low_critic_info = self.low_critic_loss(batch, grad_params)
        for k, v in low_critic_info.items():
            info[f"low_critic/{k}"] = v

        high_actor_loss, high_actor_info = self.high_actor_loss(
            batch, grad_params, high_actor_rng
        )
        for k, v in high_actor_info.items():
            info[f"high_actor/{k}"] = v

        low_actor_loss, low_actor_info = self.low_actor_loss(
            batch, grad_params, low_actor_rng
        )
        for k, v in low_actor_info.items():
            info[f"low_actor/{k}"] = v

        loss = (
            high_value_loss
            + high_critic_loss
            + low_value_loss
            + low_critic_loss
            + high_actor_loss
            + low_actor_loss
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
        """Update the agent."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, "low_critic")
        self.target_update(new_network, "high_value")

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
            observations,
            goals,
            temperature=temperature,
        )
        option = high_dist.sample(seed=high_seed)
        if self.config["option_diff"]:
            option = soft_length_normalize(option)
        else:
            option = length_normalize(option)

        low_dist = self.network.select("low_actor")(
            observations,
            option,
            temperature=temperature,
        )
        actions = low_dist.sample(seed=low_seed)
        actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_batch,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_batch: Example batch.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_observations = ex_batch["observations"]
        ex_actions = ex_batch["actions"]
        ex_goal_reps = ex_batch["high_value_goals"]
        ex_goals = ex_batch["high_value_goals"]
        action_dim = ex_actions.shape[-1]
        goal_dim = ex_goals.shape[-1]

        high_action_dim = config["rep_dim"]
        ex_options = jnp.zeros(shape=(1, config["rep_dim"]))

        # --- define networks ---
        high_value_def = GCValue(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            num_ensembles=2,
        )
        high_critic_def = GCValue(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            num_ensembles=2,
        )
        low_value_def = GCValue(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            num_ensembles=1,
        )
        low_critic_def = GCValue(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            num_ensembles=2,
        )

        low_actor_def = GCActor(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=action_dim,
            state_dependent_std=False,
            const_std=config["actor_const_std"],
        )
        high_actor_def = GCActor(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=high_action_dim,
            state_dependent_std=False,
            const_std=config["actor_const_std"],
        )
        if config["option_diff"]:
            option_rep_def = HGCEncoder(
                hidden_dims=config["rep_hidden_dims"],
                latent_dim=config["rep_dim"],
                layer_norm=config["layer_norm"],
                final_goal_layer="soft_length_normalize",
            )
        else: 
            option_rep_def = HGCEncoder(
                hidden_dims=config["rep_hidden_dims"],
                latent_dim=config["rep_dim"],
                layer_norm=config["layer_norm"],
                final_goal_layer="length_normalize",
            )

        network_info = dict[str, tuple[GCActor, tuple]](
            high_value=(high_value_def, (ex_observations, ex_goal_reps)),
            target_high_value=(
                copy.deepcopy(high_value_def),
                (ex_observations, ex_goal_reps),
            ),
            high_critic=(high_critic_def, (ex_observations, ex_goal_reps, ex_options)),
            low_critic=(low_critic_def, (ex_observations, ex_goals, ex_actions)),
            target_low_critic=(
                copy.deepcopy(low_critic_def),
                (ex_observations, ex_goals, ex_actions),
            ),
            high_actor=(high_actor_def, (ex_observations, ex_goals)),
            low_actor=(low_actor_def, (ex_observations, ex_options)),
        )
        if config["value_diff"]:
            network_info.update(
                low_value=(low_value_def, (ex_goals - ex_observations,)),
            )
        else:
            network_info.update(
                low_value=(low_value_def, (ex_observations, ex_goals)),
            )
        if config["option_diff"]:
            network_info.update(
                option_rep=(option_rep_def, {"goals": ex_goals - ex_observations}),
            )
        else:
            network_info.update(
                option_rep=(option_rep_def, {"goals": jnp.concatenate([ex_observations, ex_goals], axis=-1)}),
            )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params["modules_target_high_value"] = params["modules_high_value"]
        params["modules_target_low_critic"] = params["modules_low_critic"]

        config["action_dim"] = action_dim
        config["goal_dim"] = goal_dim
        config["low_discount"] = 1 - 1 / config["subgoal_steps"]
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # --- environment hyperparameters. ---
            action_dim=ml_collections.config_dict.placeholder(
                int
            ),  # Action dimension (set automatically).
            goal_dim=ml_collections.config_dict.placeholder(
                int
            ),  # Goal dimension (set automatically).
            # --- agent hyperparameters. ---
            agent_name="arl",  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            layer_norm=True,  # Whether to use layer normalization.
            # --- representation hyperparameters. ---
            rep_dim=10,  # Option representation dimension.
            rep_hidden_dims=(
                512,
                512,
                512,
            ),  # Representation network hidden dimensions.
            # --- value function hyperparameters. ---
            value_diff=False,  # Whether to use the difference g - s in the value function rather than separate s, g inputs.
            critic_diff=False,  # Whether to use the difference g - s in the critic rather than just the goal.
            option_diff=False,  # Whether to use the difference g - s in the option representation rather than just the goal.
            value_hidden_dims=(
                1024,
                1024,
                1024,
                1024,
            ),  # Value network hidden dimensions.
            discount=0.995,  # Discount factor.
            low_discount=ml_collections.config_dict.placeholder(
                float
            ),  # Low-level discount (set automatically).
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
            low_actor_loss="awr",  # Low-level actor loss type ("awr" or "ddpgbc").
            low_awr_alpha=3.0,  # Low-level AWR temperature.
            low_awr_clip=100.0,  # Low-level AWR clip.
            low_ddpgbc_alpha=0.1,  # Low-level DDPG+BC temperature.
            high_actor_loss="awr",  # High-level actor loss type ("awr" or "ddpgbc").
            high_awr_alpha=3.0,  # High-level AWR temperature.
            high_awr_clip=100.0,  # High-level AWR clip.
            high_ddpgbc_alpha=0.1,  # High-level DDPG+BC temperature.
            # --- dataset hyperparameters. ---
            dataset_class="HGCDataset",  # Dataset class name.
            subgoal_steps=25,  # Subgoal steps.
            value_p_curgoal=0.2,  # Probability of using the current state as the value goal.
            value_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.3,  # Probability of using a random state as the value goal.
            value_geom_sample=False,  # Whether to use geometric sampling for future value goals.
            low_value_p_curgoal=0.10,  # Probability of using the current state as the value goal.
            low_value_p_trajgoal=0.85,  # Probability of using a future state in the same trajectory as the value goal.
            low_value_p_randomgoal=0.05,  # Probability of using a random state as the value goal.
            low_value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.5,  # Probability of using a random state as the actor goal.
            actor_geom_sample=True,  # Whether to use geometric sampling for future actor goals.
            gc_negative=True,  # Whether to use "0 if s == g else -1" (True) or "1 if s == g else 0" (False) as reward.
        )
    )
    return config
