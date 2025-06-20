"""
Run Medical Diagnosis Chatbot with Hot Words Support
---------------------------------------------------
Command-line script to start the Medical Diagnosis Chatbot with enhanced
medical vocabulary support for improved speech recognition accuracy.
"""

import sys
import os
import argparse
from src.core.medical_chatbot import MedicalDiagnosisChatbot
from src.llm.language_model import StreamingLanguageModel
def show_hotwords_info():
    """Display information about medical hot words functionality"""
    print("\n🔥 医疗热词功能说明:")
    print("-" * 40)
    print("✓ 自动加载 133+ 个医疗专业术语")
    print("✓ 提升语音识别准确性")
    print("✓ 支持中医和西医术语")
    print("✓ 智能权重分配系统")
    print()

    print("🏷️ 热词分类:")
    print("  • 核心症状词 (权重5): 发烧、头痛、咳嗽、高血压等")
    print("  • 专业术语 (权重4): 胸闷、心悸、腹胀、气虚等")
    print("  • 时间描述 (权重3): 最近、经常、偶尔等")
    print()

    print("🎯 适用场景:")
    print("  • 医疗问诊对话")
    print("  • 中医症状采集")
    print("  • 病史信息录入")
    print("  • 专业术语识别")

def check_hotwords_requirements():
    """Check if hot words functionality is available"""
    try:
        from src.asr.medical_vocabulary import MedicalVocabularyManager
        vocab_manager = MedicalVocabularyManager()
        
        # Check if API key is available
        api_key = vocab_manager._ensure_api_key()
        if not api_key:
            print(f"⚠️ 医疗热词功能需要 DashScope API Key")
            print("   请在 .env 文件中设置 DASHSCOPE_API_KEY")
            print("   将使用基础语音识别功能")
            return False
            
        stats = vocab_manager.get_vocabulary_statistics()
        print(f"✅ 医疗热词模块已加载")
        print(f"   可用热词数量: {stats['total_words']}")
        print(f"   目标模型: {stats['target_model']}")
        print(f"   API Key: {'已配置' if api_key else '未配置'}")
        return True
    except Exception as e:
        print(f"⚠️ 医疗热词模块加载失败: {e}")
        print("   将使用基础语音识别功能")
        return False

def main():
    """Run the medical diagnosis chatbot with hot words support"""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run the Medical Diagnosis Chatbot with Medical Hot Words Support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python run_medical_chatbot.py                    # 使用默认设置启动
  python run_medical_chatbot.py --show-hotwords   # 显示热词信息
  python run_medical_chatbot.py --language en     # 使用英文模式
  python run_medical_chatbot.py --use-emotion     # 启用情感检测

注意事项:
  - 需要设置 ALIBABA_API_KEY 或 DASHSCOPE_API_KEY 环境变量
  - 确保麦克风权限已授权
  - 建议在安静环境中使用
        """
    )
    system_prompt = (
        "你是一个有用的、能够理解情感的语音交互助手。"
        "请根据用户的情感状态，提供适合的回答。"
        "保持回应简短但信息丰富和有帮助。"
    )
    llm = StreamingLanguageModel(
        model_name="qwen-max-latest",
        temperature=0.9,
        system_prompt=system_prompt
    )
    parser.add_argument('--language', '-l', default='zh-cn',
                        help='Language code (default: zh-cn)')
    parser.add_argument('--use-emotion', '-e', action='store_true',
                        help='Use emotion detection in responses')
    parser.add_argument('--exit-phrase', '-x', default='结束问诊',
                        help='Phrase to exit the consultation (default: 结束问诊)')
    parser.add_argument('--show-hotwords', '-s', action='store_true',
                        help='Show medical hot words information and exit')
    parser.add_argument('--no-hotwords', '-n', action='store_true',
                        help='Disable medical hot words enhancement')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output')

    args = parser.parse_args()

    # Show hot words information if requested
    if args.show_hotwords:
        show_hotwords_info()
        return

    try:
        print("🏥 医疗问诊机器人 v2.0 (医疗热词增强版)")
        print("=" * 60)

        # Check hot words availability
        hotwords_available = check_hotwords_requirements()

        if args.no_hotwords:
            print("⚠️ 用户禁用了医疗热词功能")
            hotwords_available = False

        print("\n🚀 系统初始化中...")
        print("-" * 30)

        if hotwords_available:
            print("✓ 医疗热词增强: 已启用")
        else:
            print("✓ 基础语音识别: 已启用")

        print(f"✓ 语言设置: {args.language}")
        print(f"✓ 情感检测: {'已启用' if args.use_emotion else '已禁用'}")
        print(f"✓ 退出指令: '{args.exit_phrase}'")

        if args.verbose:
            print("✓ 详细输出: 已启用")

        print("\n" + "=" * 60)

        # Create and initialize the chatbot
        # Note: The MedicalDiagnosisChatbot automatically uses enhanced ASR with hot words
        # unless there's an error, in which case it falls back to base ASR
        chatbot = MedicalDiagnosisChatbot(
            language=args.language,
            use_emotion=args.use_emotion
        )

        # Display hot words status after initialization
        if hasattr(chatbot.recognizer, 'vocabulary_manager'):
            vocab_stats = chatbot.recognizer.get_vocabulary_statistics()
            if vocab_stats.get('vocabulary_id'):
                vocab_status = "复用" if vocab_stats.get('is_reused', False) else "新建"
                print(f"🎯 医疗热词表已就绪 ({vocab_status}): {vocab_stats['vocabulary_id']}")
            else:
                print("⚠️ 医疗热词表创建失败，使用基础识别")

        print("\n💬 问诊提示:")
        print("  • 请清晰地描述您的症状")
        print("  • 医疗热词功能会自动提升术语识别准确性")
        print("  • 支持中医术语如：气虚、血虚、湿热等")
        print(f"  • 说 '{args.exit_phrase}' 可以随时结束问诊")
        print("\n" + "=" * 60)

        # Run the consultation
        summary = chatbot.run_consultation(exit_phrase=args.exit_phrase)

        # Save the consultation summary to a file
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"medical_consultation_{timestamp}.md"

        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"# 医疗问诊记录\n\n")
            f.write(f"**时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**语言**: {args.language}\n")
            f.write(f"**热词增强**: {'是' if hotwords_available else '否'}\n")
            f.write(f"**情感检测**: {'是' if args.use_emotion else '否'}\n\n")
            f.write("---\n\n")
            f.write(summary)
            f.write("---\n\n")
            summary_prompt = f"""你是一位专业的中医，需要根据以下医患对话生成病情总结：

                            对话记录：
                            {summary}

                            总结要求：
                            1. 只提取患者实际存在的症状
                            2. 用专业但易懂的语言描述
                            3. 按照症状重要性排序
                            4. 忽略患者否认的症状
                            5. 给出患者可能患的病
                            6. 格式示例：
                            患者主诉：[主要症状]
                            伴随症状：[次要症状]
                            其他情况：[其他信息]
                            可能病症：[可能患病]   
                            中医症状总结：
                            尽可能根据内容多分析一点内容出来，包括病因，可能的病症，以及解决方法
"""

            summary_result = llm.generate_response(
                user_input=summary_prompt,
                conversation_history=[]
            )
            natural_summary = summary_result["response"] if summary_result["success"] else "无法生成自然语言总结"
            f.write(natural_summary)
        print(f"\n📄 问诊摘要已保存至文件: {filename}")
        print("=" * 60)
        
        # Show final statistics
        if hasattr(chatbot.recognizer, 'get_vocabulary_statistics'):
            try:
                final_stats = chatbot.recognizer.get_vocabulary_statistics()
                print(f"\n📊 本次问诊统计:")
                print(f"   热词总数: {final_stats['total_words']}")
                if final_stats.get('vocabulary_id'):
                    print(f"   热词表ID: {final_stats['vocabulary_id']}")
            except:
                pass
        
    except KeyboardInterrupt:
        print("\n\n👋 用户中断，正在安全退出...")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
    finally:
        try:
            if 'chatbot' in locals():
                print("\n🧹 正在清理资源...")
                chatbot.cleanup()
                print("✓ 资源清理完成")
        except Exception as e:
            if args.verbose:
                print(f"清理过程中出现警告: {e}")
        print("\n🏥 医疗问诊机器人已安全关闭。")
        print("感谢使用！祝您健康！")

if __name__ == "__main__":
    main()