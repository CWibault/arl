# pointmaze-giant-navigate-v0 (gciql)
for s in {0..3}; do
    python3.10 main.py --env_name=pointmaze-giant-navigate-v0 --agent=ogb_impls/agents/gciql.py --agent.actor_loss=ddpgbc --agent.ddpgbc_alpha=0.1 --agent.discount=0.995 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done
# pointmaze-giant-navigate-v0 (hiql1v)
for s in {0..3}; do
    python3.10 main.py --env_name=pointmaze-giant-navigate-v0 --agent=ogb_impls/agents/hiql1v.py --agent.discount=0.995 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done
# pointmaze-giant-navigate-v0 (hiql2v)
for s in {0..3}; do
    python3.10 main.py --env_name=pointmaze-giant-navigate-v0 --agent=ogb_impls/agents/hiql2v.py --agent.use_hiql2v_rep=True --agent.low_actor_grad=False --agent.discount=0.995 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done
# pointmaze-giant-navigate-v0 (arl)
for s in {0..3}; do
    python3.10 main.py --env_name=pointmaze-giant-navigate-v0 --agent=ogb_impls/agents/arl.py --agent.discount=0.995 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done

# pointmaze-giant-stitch-v0 (gciql)
for s in {0..3}; do
    python3.10 main.py --env_name=pointmaze-giant-stitch-v0 --agent=ogb_impls/agents/gciql.py --agent.actor_loss=ddpgbc --agent.ddpgbc_alpha=0.1 --agent.discount=0.995 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done
# pointmaze-giant-stitch-v0 (hiql1v)
for s in {0..3}; do
    python3.10 main.py --env_name=pointmaze-giant-stitch-v0 --agent=ogb_impls/agents/hiql1v.py --agent.discount=0.995 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done
# pointmaze-giant-stitch-v0 (hiql2v)
for s in {0..3}; do
    python3.10 main.py --env_name=pointmaze-giant-stitch-v0 --agent=ogb_impls/agents/hiql2v.py --agent.use_hiql2v_rep=True --agent.low_actor_grad=False --agent.discount=0.995 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done
# pointmaze-giant-stitch-v0 (arl)
for s in {0..3}; do
    python3.10 main.py --env_name=pointmaze-giant-stitch-v0 --agent=ogb_impls/agents/arl.py --agent.discount=0.995 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done

# antmaze-teleport-stitch-v0 (gciql)
for s in {0..3}; do
    python3.10 main.py --env_name=antmaze-teleport-stitch-v0 --agent=ogb_impls/agents/gciql.py --agent.actor_loss=ddpgbc --agent.ddpgbc_alpha=0.1 --agent.discount=0.99 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done
# antmaze-teleport-stitch-v0 (hiql1v)
for s in {0..3}; do
    python3.10 main.py --env_name=antmaze-teleport-stitch-v0 --agent=ogb_impls/agents/hiql1v.py --agent.discount=0.99 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done
# antmaze-teleport-stitch-v0 (hiql2v)
for s in {0..3}; do
    python3.10 main.py --env_name=antmaze-teleport-stitch-v0 --agent=ogb_impls/agents/hiql2v.py --agent.use_hiql2v_rep=True --agent.low_actor_grad=False --agent.discount=0.99 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done
# antmaze-teleport-stitch-v0 (arl)
for s in {0..3}; do
    python3.10 main.py --env_name=antmaze-teleport-stitch-v0 --agent=ogb_impls/agents/arl.py --agent.discount=0.99 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done

# antmaze-giant-navigate-v0 (gciql)
for s in {0..3}; do
    python3.10 main.py --env_name=antmaze-giant-navigate-v0 --agent=ogb_impls/agents/gciql.py --agent.actor_loss=ddpgbc --agent.ddpgbc_alpha=0.1 --agent.discount=0.995 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done
# antmaze-giant-navigate-v0 (hiql1v)
for s in {0..3}; do
    python3.10 main.py --env_name=antmaze-giant-navigate-v0 --agent=ogb_impls/agents/hiql1v.py --agent.discount=0.995 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done
# antmaze-giant-navigate-v0 (hiql2v)
for s in {0..3}; do
    python3.10 main.py --env_name=antmaze-giant-navigate-v0 --agent=ogb_impls/agents/hiql2v.py --agent.use_hiql2v_rep=True --agent.low_actor_grad=False --agent.discount=0.995 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done
# antmaze-giant-navigate-v0 (arl)
for s in {0..3}; do
    python3.10 main.py --env_name=antmaze-giant-navigate-v0 --agent=ogb_impls/agents/arl.py --agent.discount=0.995 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done

# antmaze-giant-stitch-v0 (gciql)
for s in {0..3}; do
    python3.10 main.py --env_name=antmaze-giant-stitch-v0 --agent=ogb_impls/agents/gciql.py --agent.actor_loss=ddpgbc --agent.ddpgbc_alpha=0.1 --agent.discount=0.995 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done
# antmaze-giant-stitch-v0 (hiql1v)
for s in {0..3}; do
    python3.10 main.py --env_name=antmaze-giant-stitch-v0 --agent=ogb_impls/agents/hiql1v.py --agent.discount=0.995 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done
# antmaze-giant-stitch-v0 (hiql2v)
for s in {0..3}; do
    python3.10 main.py --env_name=antmaze-giant-stitch-v0 --agent=ogb_impls/agents/hiql2v.py --agent.use_hiql2v_rep=True --agent.low_actor_grad=False --agent.discount=0.995 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done
# antmaze-giant-stitch-v0 (arl)
for s in {0..3}; do
    python3.10 main.py --env_name=antmaze-giant-stitch-v0 --agent=ogb_impls/agents/arl.py --agent.discount=0.995 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done

# humanoidmaze-giant-navigate-v0 (gciql)
for s in {0..3}; do
    python3.10 main.py --env_name=humanoidmaze-giant-navigate-v0 --agent=ogb_impls/agents/gciql.py --agent.actor_loss=ddpgbc --agent.ddpgbc_alpha=0.1 --agent.discount=0.999 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done
# humanoidmaze-giant-navigate-v0 (hiql1v)
for s in {0..3}; do
    python3.10 main.py --env_name=humanoidmaze-giant-navigate-v0 --agent=ogb_impls/agents/hiql1v.py --agent.discount=0.999 --agent.subgoal_steps=100 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done
# humanoidmaze-giant-navigate-v0 (hiql2v)
for s in {0..3}; do
    python3.10 main.py --env_name=humanoidmaze-giant-navigate-v0 --agent=ogb_impls/agents/hiql2v.py --agent.use_hiql2v_rep=True --agent.low_actor_grad=False --agent.discount=0.999 --agent.subgoal_steps=100 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done
# humanoidmaze-giant-navigate-v0 (arl)
for s in {0..3}; do
    python3.10 main.py --env_name=humanoidmaze-giant-navigate-v0 --agent=ogb_impls/agents/arl.py --agent.discount=0.999 --agent.subgoal_steps=100 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done

# humanoidmaze-giant-stitch-v0 (gciql)
for s in {0..3}; do
    python3.10 main.py --env_name=humanoidmaze-giant-stitch-v0 --agent=ogb_impls/agents/gciql.py --agent.actor_loss=ddpgbc --agent.ddpgbc_alpha=0.1 --agent.discount=0.999 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done
# humanoidmaze-giant-stitch-v0 (hiql1v)
for s in {0..3}; do
    python3.10 main.py --env_name=humanoidmaze-giant-stitch-v0 --agent=ogb_impls/agents/hiql1v.py --agent.discount=0.999 --agent.subgoal_steps=100 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done
# humanoidmaze-giant-stitch-v0 (hiql2v)
for s in {0..3}; do
    python3.10 main.py --env_name=humanoidmaze-giant-stitch-v0 --agent=ogb_impls/agents/hiql2v.py --agent.use_hiql2v_rep=True --agent.low_actor_grad=False --agent.discount=0.999 --agent.subgoal_steps=100 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done
# humanoidmaze-giant-stitch-v0 (arl)
for s in {0..3}; do
    python3.10 main.py --env_name=humanoidmaze-giant-stitch-v0 --agent=ogb_impls/agents/arl.py --agent.discount=0.999 --agent.subgoal_steps=100 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done

# cube-double-play-v0 (gciql)
for s in {0..3}; do
    python3.10 main.py --env_name=cube-double-play-v0 --agent=ogb_impls/agents/gciql.py --agent.actor_loss=ddpgbc --agent.ddpgbc_alpha=1.0 --agent.discount=0.99 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done
# cube-double-play-v0 (hiql1v)
for s in {0..3}; do
    python3.10 main.py --env_name=cube-double-play-v0 --agent=ogb_impls/agents/hiql1v.py --agent.high_actor_loss=ddpgbc --agent.discount=0.99 --agent.subgoal_steps=25 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done
# cube-double-play-v0 (hiql2v)
for s in {0..3}; do
    python3.10 main.py --env_name=cube-double-play-v0 --agent=ogb_impls/agents/hiql2v.py --agent.use_hiql2v_rep=True --agent.low_actor_grad=False --agent.high_actor_loss=ddpgbc --agent.discount=0.99 --agent.subgoal_steps=25 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done
# cube-double-play-v0 (arl)
for s in {0..3}; do
    python3.10 main.py --env_name=cube-double-play-v0 --agent=ogb_impls/agents/arl.py --agent.high_actor_loss=ddpgbc --agent.discount=0.99 --agent.subgoal_steps=25 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done

# cube-triple-play-v0 (gciql)
for s in {0..3}; do
    python3.10 main.py --env_name=cube-triple-play-v0 --agent=ogb_impls/agents/gciql.py --agent.actor_loss=ddpgbc --agent.ddpgbc_alpha=1.0 --agent.discount=0.99 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done
# cube-triple-play-v0 (hiql1v)
for s in {0..3}; do
    python3.10 main.py --env_name=cube-triple-play-v0 --agent=ogb_impls/agents/hiql1v.py --agent.high_actor_loss=ddpgbc --agent.discount=0.99 --agent.subgoal_steps=25 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done
# cube-triple-play-v0 (hiql2v)
for s in {0..3}; do
    python3.10 main.py --env_name=cube-triple-play-v0 --agent=ogb_impls/agents/hiql2v.py --agent.use_hiql2v_rep=True --agent.low_actor_grad=False --agent.high_actor_loss=ddpgbc --agent.discount=0.99 --agent.subgoal_steps=25 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done
# cube-triple-play-v0 (arl)
for s in {0..3}; do
    python3.10 main.py --env_name=cube-triple-play-v0 --agent=ogb_impls/agents/arl.py --agent.high_actor_loss=ddpgbc --agent.discount=0.99 --agent.subgoal_steps=25 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done

# cube-quadruple-play-v0 (gciql)
for s in {0..3}; do
    python3.10 main.py --env_name=cube-quadruple-play-v0 --agent=ogb_impls/agents/gciql.py --agent.actor_loss=ddpgbc --agent.ddpgbc_alpha=1.0 --agent.discount=0.99 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done
# cube-quadruple-play-v0 (hiql1v)
for s in {0..3}; do
    python3.10 main.py --env_name=cube-quadruple-play-v0 --agent=ogb_impls/agents/hiql1v.py --agent.high_actor_loss=ddpgbc --agent.discount=0.99 --agent.subgoal_steps=25 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done
# cube-quadruple-play-v0 (hiql2v)
for s in {0..3}; do
    python3.10 main.py --env_name=cube-quadruple-play-v0 --agent=ogb_impls/agents/hiql2v.py --agent.use_hiql2v_rep=True --agent.low_actor_grad=False --agent.high_actor_loss=ddpgbc --agent.discount=0.99 --agent.subgoal_steps=25 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done
# cube-quadruple-play-v0 (arl)
for s in {0..3}; do
    python3.10 main.py --env_name=cube-quadruple-play-v0 --agent=ogb_impls/agents/arl.py --agent.high_actor_loss=ddpgbc --agent.discount=0.99 --agent.subgoal_steps=25 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done

# puzzle-3x3-play-v0 (gciql)
for s in {0..3}; do
    python3.10 main.py --env_name=puzzle-3x3-play-v0 --agent=ogb_impls/agents/gciql.py --agent.actor_loss=ddpgbc --agent.ddpgbc_alpha=1.0 --agent.discount=0.99 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done
# puzzle-3x3-play-v0 (hiql1v)
for s in {0..3}; do
    python3.10 main.py --env_name=puzzle-3x3-play-v0 --agent=ogb_impls/agents/hiql1v.py --agent.high_actor_loss=ddpgbc --agent.discount=0.99 --agent.subgoal_steps=25 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done
# puzzle-3x3-play-v0 (hiql2v)
for s in {0..3}; do
    python3.10 main.py --env_name=puzzle-3x3-play-v0 --agent=ogb_impls/agents/hiql2v.py --agent.use_hiql2v_rep=True --agent.low_actor_grad=False --agent.high_actor_loss=ddpgbc --agent.discount=0.99 --agent.subgoal_steps=25 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done
# puzzle-3x3-play-v0 (arl)
for s in {0..3}; do
    python3.10 main.py --env_name=puzzle-3x3-play-v0 --agent=ogb_impls/agents/arl.py --agent.high_actor_loss=ddpgbc --agent.discount=0.99 --agent.subgoal_steps=25 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done

# puzzle-4x4-play-v0 (gciql)
for s in {0..3}; do
    python3.10 main.py --env_name=puzzle-4x4-play-v0 --agent=ogb_impls/agents/gciql.py --agent.actor_loss=ddpgbc --agent.ddpgbc_alpha=1.0 --agent.discount=0.99 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done
# puzzle-4x4-play-v0 (hiql1v)
for s in {0..3}; do
    python3.10 main.py --env_name=puzzle-4x4-play-v0 --agent=ogb_impls/agents/hiql1v.py --agent.high_actor_loss=ddpgbc --agent.discount=0.99 --agent.subgoal_steps=25 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done
# puzzle-4x4-play-v0 (hiql2v)
for s in {0..3}; do
    python3.10 main.py --env_name=puzzle-4x4-play-v0 --agent=ogb_impls/agents/hiql2v.py --agent.use_hiql2v_rep=True --agent.low_actor_grad=False --agent.high_actor_loss=ddpgbc --agent.discount=0.99 --agent.subgoal_steps=25 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done
# puzzle-4x4-play-v0 (arl)
for s in {0..3}; do
    python3.10 main.py --env_name=puzzle-4x4-play-v0 --agent=ogb_impls/agents/arl.py --agent.high_actor_loss=ddpgbc --agent.discount=0.99 --agent.subgoal_steps=25 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done

# puzzle-4x5-play-v0 (gciql)
for s in {0..3}; do
    python3.10 main.py --env_name=puzzle-4x5-play-v0 --agent=ogb_impls/agents/gciql.py --agent.actor_loss=ddpgbc --agent.ddpgbc_alpha=1.0 --agent.discount=0.99 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done
# puzzle-4x5-play-v0 (hiql1v)
for s in {0..3}; do
    python3.10 main.py --env_name=puzzle-4x5-play-v0 --agent=ogb_impls/agents/hiql1v.py --agent.high_actor_loss=ddpgbc --agent.discount=0.99 --agent.subgoal_steps=25 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done
# puzzle-4x5-play-v0 (hiql2v)
for s in {0..3}; do
    python3.10 main.py --env_name=puzzle-4x5-play-v0 --agent=ogb_impls/agents/hiql2v.py --agent.use_hiql2v_rep=True --agent.low_actor_grad=False --agent.high_actor_loss=ddpgbc --agent.discount=0.99 --agent.subgoal_steps=25 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done
# puzzle-4x5-play-v0 (arl)
for s in {0..3}; do
    python3.10 main.py --env_name=puzzle-4x5-play-v0 --agent=ogb_impls/agents/arl.py --agent.high_actor_loss=ddpgbc --agent.discount=0.99 --agent.subgoal_steps=25 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done

# puzzle-4x6-play-v0 (gciql)
for s in {0..3}; do
    python3.10 main.py --env_name=puzzle-4x6-play-v0 --agent=ogb_impls/agents/gciql.py --agent.actor_loss=ddpgbc --agent.ddpgbc_alpha=1.0 --agent.discount=0.99 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done
# puzzle-4x6-play-v0 (hiql1v)
for s in {0..3}; do
    python3.10 main.py --env_name=puzzle-4x6-play-v0 --agent=ogb_impls/agents/hiql1v.py --agent.high_actor_loss=ddpgbc --agent.discount=0.99 --agent.subgoal_steps=25 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done
# puzzle-4x6-play-v0 (hiql2v)
for s in {0..3}; do
    python3.10 main.py --env_name=puzzle-4x6-play-v0 --agent=ogb_impls/agents/hiql2v.py --agent.use_hiql2v_rep=True --agent.low_actor_grad=False --agent.high_actor_loss=ddpgbc --agent.discount=0.99 --agent.subgoal_steps=25 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done
# puzzle-4x6-play-v0 (arl)
for s in {0..3}; do
    python3.10 main.py --env_name=puzzle-4x6-play-v0 --agent=ogb_impls/agents/arl.py --agent.high_actor_loss=ddpgbc --agent.discount=0.99 --agent.subgoal_steps=25 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done

# scene-play-v0 (gciql)
for s in {0..3}; do
    python3.10 main.py --env_name=scene-play-v0 --agent=ogb_impls/agents/gciql.py --agent.actor_loss=ddpgbc --agent.ddpgbc_alpha=1.0 --agent.discount=0.99 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done
# scene-play-v0 (hiql1v)
for s in {0..3}; do
    python3.10 main.py --env_name=scene-play-v0 --agent=ogb_impls/agents/hiql1v.py --agent.high_actor_loss=awr --agent.discount=0.99 --agent.subgoal_steps=25 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done
# scene-play-v0 (hiql2v)
for s in {0..3}; do
    python3.10 main.py --env_name=scene-play-v0 --agent=ogb_impls/agents/hiql2v.py --agent.use_hiql2v_rep=True --agent.low_actor_grad=False --agent.high_actor_loss=awr --agent.discount=0.99 --agent.subgoal_steps=25 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done
# scene-play-v0 (arl)
for s in {0..3}; do
    python3.10 main.py --env_name=scene-play-v0 --agent=ogb_impls/agents/arl.py --agent.high_actor_loss=awr --agent.discount=0.99 --agent.subgoal_steps=25 --eval_interval=1000000 --eval_episodes=20 --eval_on_start=False --seed=$s
done