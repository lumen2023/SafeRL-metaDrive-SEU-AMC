"""
测试BEV GIF生成功能

这个脚本用于验证周期性测试中的BEV GIF生成功能是否正常工作。
运行此脚本可以快速诊断渲染问题，而无需等待完整的训练过程。

使用方法:
    python test_bev_gif.py
"""
import gymnasium as gym
import sys
import os

# 注册环境（与train_sacl.py中相同）
gym.register(
    id="SafeMetaDrive-validation",
    entry_point="env:SafeMetaDriveEnv_mini",
    max_episode_steps=1000,
    kwargs={"config": {"num_scenarios": 50, "start_seed": 1000}},
)


def test_bev_render():
    """测试BEV渲染功能"""
    print("=" * 80)
    print("开始测试BEV GIF生成功能")
    print("=" * 80)
    
    # 创建环境
    print("\n1. 创建验证环境...")
    env = gym.make('SafeMetaDrive-validation')
    print(f"   ✓ 环境创建成功: {env}")
    
    # 重置环境
    print("\n2. 重置环境...")
    obs, info = env.reset()
    print(f"   ✓ 环境重置成功")
    print(f"   - 观察空间形状: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
    print(f"   - 初始信息: {list(info.keys()) if isinstance(info, dict) else 'N/A'}")
    
    # 检查top_down_renderer初始状态
    print("\n3. 检查渲染器初始状态...")
    print(f"   - top_down_renderer: {env.top_down_renderer}")
    
    # 首次调用render初始化top_down_renderer
    print("\n4. 初始化TopDownRenderer...")
    try:
        env.render(
            mode="topdown",
            window=False,  # 不显示窗口（离屏渲染）
            screen_record=True,  # 记录帧
            target_agent_heading_up=True,
            film_size=(2000, 2000),
            screen_size=(800, 800),
        )
        print(f"   ✓ TopDownRenderer初始化成功")
        print(f"   - Renderer类型: {type(env.top_down_renderer)}")
        print(f"   - screen_frames数量: {len(env.top_down_renderer.screen_frames) if env.top_down_renderer else 0}")
    except Exception as e:
        print(f"   ✗ 初始化失败: {e}")
        env.close()
        return False
    
    # 执行几步并收集帧
    print("\n5. 执行动作并收集BEV帧...")
    num_steps = 10
    for step in range(num_steps):
        action = env.action_space.sample()  # 随机动作
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 每次step后渲染以记录帧
        env.render(
            mode="topdown",
            window=False,
            screen_record=True,
            target_agent_heading_up=True,
        )
        
        if terminated or truncated:
            print(f"   - Episode在第{step + 1}步结束")
            break
    
    print(f"   ✓ 完成{num_steps}步仿真")
    print(f"   - 记录的帧数: {len(env.top_down_renderer.screen_frames) if env.top_down_renderer else 0}")
    
    # 生成GIF
    print("\n6. 生成BEV GIF...")
    gif_path = "test_bev_output.gif"
    try:
        if env.top_down_renderer and env.top_down_renderer.screen_frames:
            env.top_down_renderer.generate_gif(gif_path)
            print(f"   ✓ GIF生成成功: {gif_path}")
            
            # 检查文件大小
            if os.path.exists(gif_path):
                file_size = os.path.getsize(gif_path)
                print(f"   - 文件大小: {file_size / 1024:.2f} KB")
                
                if file_size > 0:
                    print(f"\n{'=' * 80}")
                    print("✅ 测试成功！BEV GIF生成功能正常工作。")
                    print(f"{'=' * 80}")
                    print(f"\n生成的GIF文件位于: {os.path.abspath(gif_path)}")
                    print("您可以使用图片查看器打开该文件进行验证。")
                    return True
                else:
                    print(f"\n{'=' * 80}")
                    print("❌ 测试失败：GIF文件为空")
                    print(f"{'=' * 80}")
                    return False
        else:
            print(f"   ✗ 没有可用的帧数据")
            print(f"   - top_down_renderer: {env.top_down_renderer}")
            print(f"   - screen_frames: {env.top_down_renderer.screen_frames if env.top_down_renderer else None}")
            return False
    except Exception as e:
        print(f"   ✗ GIF生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        env.close()


if __name__ == "__main__":
    try:
        success = test_bev_render()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
