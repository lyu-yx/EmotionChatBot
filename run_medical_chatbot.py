"""
Run Medical Diagnosis Chatbot
----------------------------
Command-line script to start the Medical Diagnosis Chatbot.
"""

import sys
import os
import argparse
from src.core.medical_chatbot import MedicalDiagnosisChatbot

def main():
    """Run the medical diagnosis chatbot"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the Medical Diagnosis Chatbot')
    parser.add_argument('--language', '-l', default='zh-cn', 
                        help='Language code (default: zh-cn)')
    parser.add_argument('--use-emotion', '-e', action='store_true',
                        help='Use emotion detection in responses')
    parser.add_argument('--exit-phrase', '-x', default='结束问诊',
                        help='Phrase to exit the consultation (default: 结束问诊)')
    
    args = parser.parse_args()
    
    try:
        print("="*50)
        print("医疗问诊机器人启动中...")
        print("="*50)
        
        # Create and initialize the chatbot
        chatbot = MedicalDiagnosisChatbot(
            language=args.language,
            use_emotion=args.use_emotion
        )
        
        # Run the consultation
        summary = chatbot.run_consultation(exit_phrase=args.exit_phrase)
        
        # Save the consultation summary to a file
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"medical_consultation_{timestamp}.md"
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(summary)
        
        print(f"\n问诊摘要已保存至文件: {filename}")
        print("="*50)
        
    except KeyboardInterrupt:
        print("\n用户中断，正在退出...")
    except Exception as e:
        print(f"错误: {e}")
    finally:
        try:
            if 'chatbot' in locals():
                chatbot.cleanup()
        except:
            pass
        print("医疗问诊机器人已关闭。")

if __name__ == "__main__":
    main()