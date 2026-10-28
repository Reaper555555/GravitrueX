"""CPU-only integration test for the hybrid rollout coordinator.

The batched tensor arena is intentionally run on CPU here because test hosts may
not have CUDA. This validates that CPU ProcessPool episodes and tensor-batch
episodes are merged into one PPO update; it is not a CUDA performance benchmark.
"""
from pathlib import Path
import sys
import tempfile
import shutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import gravitruex_ai as e
from gravitruex_gpu_batch import GPUVectorArenaBatch


def main():
    tmp = Path(tempfile.mkdtemp(prefix='grav_hybrid_test_'))
    old_model_root = e.MODEL_ROOT
    e.MODEL_ROOT = tmp / "models"  # keep test metadata out of the user's real model library
    base = e.ActorCritic(seed=12, width=32, depth=2, device_mode='cpu')
    trainer = e.Trainer(base, 1, 1, False, tmp, workers=1, opponent_mode='heuristic',
                        device_mode='cpu', batch_size=32, epochs=1,
                        hybrid_gpu_rollout=True, gpu_batch_envs=2, model_id='test', model_name='test')
    trainer.policy.save(trainer.save_path)
    original_make_task = trainer._make_task
    def make_task():
        task = original_make_task()
        task['max_decisions'] = 8
        return task
    trainer._make_task = make_task
    def collect_gpu_cpu_tensor(seed=None):
        # Simulate a slower GPU batch so CPU workers have time to refill the buffer.
        import time
        time.sleep(4.0)
        batch = GPUVectorArenaBatch(engine=e, policy=trainer.policy, count=2, blue_n=1, red_n=1,
                                    opponent_mode='heuristic', max_decisions=8, device='cpu', seed=seed or 987)
        out = batch.collect(trainer.stop_event)
        trainer.stats['test_gpu_batch_envs'] = len(out)
        return out
    trainer._collect_gpu_batch = collect_gpu_cpu_tensor
    orig_update = trainer.policy.update_ppo
    update_calls = {'n': 0}
    def update_ppo_once(records, *args, **kwargs):
        loss = orig_update(records, *args, **kwargs)
        update_calls['n'] += 1
        trainer.stop_event.set()
        if trainer.mp_stop_event is not None:
            trainer.mp_stop_event.set()
        return loss
    trainer.policy.update_ppo = update_ppo_once
    trainer.start()
    trainer.join(timeout=120)
    if trainer.is_alive():
        trainer.stop(); raise TimeoutError('trainer did not finish')
    stats = trainer.get_stats()
    print('trainer_alive', trainer.is_alive())
    print('update_calls', update_calls['n'])
    print('hybrid', stats.get('hybrid_gpu_rollout'))
    print('episodes', stats.get('episodes'))
    print('last_rollout_episodes', stats.get('rollout_episodes_last'))
    print('cpu_last', stats.get('cpu_rollout_episodes_last'))
    print('gpu_last', stats.get('gpu_rollout_episodes_last'))
    print('error', stats.get('message'))
    assert update_calls['n'] == 1, f'expected 1 PPO update, got {update_calls["n"]}'
    # 1 CPU process episode + 2 tensor-arena episodes in the same rollout batch.
    assert stats.get('episodes', 0) >= 3, f'expected merged CPU+GPU episodes, got {stats.get("episodes")}'
    assert stats.get('cpu_rollout_episodes_last', 0) >= 1
    assert stats.get('gpu_rollout_episodes_last', 0) == 2
    assert stats.get('cpu_episodes_during_gpu', 0) >= 2, "CPU should collect more than one episode while GPU producer is busy"
    assert stats.get('cpu_rollout_sec_last', 0) > 0, "CPU stage timer missing"
    assert stats.get('gpu_rollout_sec_last', 0) > 0, "GPU/tensor stage timer missing"
    assert len(stats.get('last_two_cycles', [])) >= 1, "Last-cycle log missing"
    assert (tmp / 'training_report.txt').exists(), "Text report was not written"
    assert (tmp / 'training_metrics.jsonl').exists(), "JSONL cycle log was not written"
    report = (tmp / 'training_report.txt').read_text(encoding='utf-8')
    assert 'CPU:' in report and 'GPU:' in report and 'PPO:' in report
    print('HYBRID ORCHESTRATION + SEPARATE TIMINGS + REPORT TEST PASSED (CPU tensor backend; CUDA throughput not measured)')
    e.MODEL_ROOT = old_model_root
    shutil.rmtree(tmp, ignore_errors=True)

if __name__ == '__main__':
    main()
