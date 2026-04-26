import torch

def simple_convert():
    # --- 配置路径 ---
    input_path = 'models\cartpole_ppo.torch'  # 你的源文件名
    output_path = 'runs\cartpole_ppo.pt'    # 你想要的目标文件名
    
    print(f"正在加载 {input_path} ...")
    
    try:
        # 1. 加载文件
        # map_location='cpu' 确保即使你在没有 GPU 的机器上也能转换 GPU 训练的模型
        content = torch.load(input_path, map_location='cpu', weights_only=False)
        
        # 2. 另存为 .pt
        torch.save(content, output_path)
        
        print(f"✅ 转换成功！")
        print(f"源文件内容类型: {type(content)}")
        print(f"已保存为: {output_path}")
        
    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {input_path}")
    except Exception as e:
        print(f"❌ 发生错误: {e}")

if __name__ == '__main__':
    simple_convert()