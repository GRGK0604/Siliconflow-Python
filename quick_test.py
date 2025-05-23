#!/usr/bin/env python3
import requests
import json
import time

def format_time(seconds):
    """格式化时间显示"""
    if seconds is None:
        return "未知"
    if seconds < 60:
        return f"{int(seconds)} 秒"
    elif seconds < 3600:
        return f"{int(seconds // 60)} 分 {int(seconds % 60)} 秒"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours} 小时 {minutes} 分"

def test_health():
    try:
        response = requests.get("http://localhost:7898/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("=== 系统健康状态 ===")
            print(f"状态: {data.get('status')}")
            print(f"数据库: {data.get('database')}")
            print(f"API密钥: {data.get('api_keys')} ({data.get('api_keys_count', 0)} 个)")
            
            auto_refresh = data.get('auto_refresh', {})
            print(f"\n=== 自动刷新状态 ===")
            print(f"启用: {auto_refresh.get('enabled')}")
            print(f"运行中: {auto_refresh.get('running')}")
            print(f"运行次数: {auto_refresh.get('run_count')}")
            print(f"错误次数: {auto_refresh.get('error_count')}")
            
            last_run = auto_refresh.get('last_run_time')
            if last_run and last_run > 0:
                print(f"上次运行: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last_run))}")
                time_since = auto_refresh.get('time_since_last_run')
                if time_since is not None:
                    print(f"距离上次运行: {format_time(time_since)}")
                
                next_run = auto_refresh.get('next_run_in')
                if next_run is not None:
                    print(f"下次运行倒计时: {format_time(next_run)}")
            else:
                print("上次运行: 尚未运行")
            
            # 显示性能指标
            performance = auto_refresh.get('performance', {})
            if performance:
                print(f"\n=== 性能指标 ===")
                print(f"总处理密钥数: {performance.get('total_keys_processed', 0)}")
                print(f"总更新密钥数: {performance.get('total_keys_updated', 0)}")
                print(f"总移除密钥数: {performance.get('total_keys_removed', 0)}")
                print(f"平均处理时间: {performance.get('average_processing_time', 0)} 秒")
                print(f"成功率: {performance.get('success_rate', 0)}%")
            
            # 显示最后一次批处理统计
            last_batch = auto_refresh.get('last_batch', {})
            if last_batch:
                print(f"\n=== 最后一次批处理 ===")
                print(f"处理密钥数: {last_batch.get('processed', 0)}")
                print(f"更新: {last_batch.get('updated', 0)}")
                print(f"移除: {last_batch.get('removed', 0)}")
                print(f"错误: {last_batch.get('errors', 0)}")
                print(f"处理时间: {last_batch.get('processing_time', 0):.2f} 秒")
                
                batch_time = last_batch.get('timestamp')
                if batch_time:
                    print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(batch_time))}")
            
            last_error = auto_refresh.get('last_error')
            if last_error:
                print(f"\n❌ 最后错误: {last_error}")
            
            return True
        else:
            print(f"健康检查失败，状态码: {response.status_code}")
            return False
    except Exception as e:
        print(f"连接失败: {e}")
        return False

def test_performance_summary():
    """显示性能摘要"""
    try:
        response = requests.get("http://localhost:7898/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            auto_refresh = data.get('auto_refresh', {})
            performance = auto_refresh.get('performance', {})
            
            print("\n" + "="*60)
            print("📊 性能摘要")
            print("="*60)
            
            if auto_refresh.get('enabled'):
                if auto_refresh.get('running'):
                    print("🟢 自动刷新: 运行中")
                else:
                    print("🔴 自动刷新: 已停止")
                
                run_count = auto_refresh.get('run_count', 0)
                error_count = auto_refresh.get('error_count', 0)
                if run_count > 0:
                    error_rate = (error_count / run_count) * 100
                    print(f"📈 运行统计: {run_count} 次运行, {error_count} 次错误 ({error_rate:.1f}%)")
                
                success_rate = performance.get('success_rate', 0)
                if success_rate >= 95:
                    status_icon = "🟢"
                elif success_rate >= 80:
                    status_icon = "🟡"
                else:
                    status_icon = "🔴"
                print(f"{status_icon} 成功率: {success_rate}%")
                
                avg_time = performance.get('average_processing_time', 0)
                if avg_time > 0:
                    print(f"⏱️  平均处理时间: {avg_time} 秒")
                
            else:
                print("⚪ 自动刷新: 已禁用")
            
    except Exception as e:
        print(f"获取性能数据失败: {e}")

if __name__ == "__main__":
    test_health()
    test_performance_summary() 