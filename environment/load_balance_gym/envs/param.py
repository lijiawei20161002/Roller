import argparse 
import pandas as pd

parser = argparse.ArgumentParser(description="parameters")

# -- Basic --
parser.add_argument('--seed', type=int, default=10, help='random seed (default: 42)')
parser.add_argument('--eps', type=float, default=1e-6, help='epsilon (default: 1e-6)')
parser.add_argument('--logging_level', type=str, default='info', help='logging level (default: info)')
parser.add_argument('--log_to', type=str, default='print', help='logging destination, "print" or a filepath (default: print)')

# -- Load Balance --
parser.add_argument('--num_servers', type=int, default=73, help='number of servers (default: 10)')
#parser.add_argument('--num_servers', type=int, default=10, help='number of servers (default: 10)')
parser.add_argument('--num_stream_jobs', type=int, default=100, help='number of streaming jobs (default: 1000)')
parser.add_argument('--service_rates', type=float, default=[10] * 73, #(0.15 + (pd.Series(range(100)))*0.1).tolist(), 
                    nargs='+', help='workers service rates (default: [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05]')
#parser.add_argument('--service_rates', type=float, default=[0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05], 
                    #nargs='+', help='workers service rates (default: [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05]')
parser.add_argument('--job_interval', type=int, default=4, help='job arrival interval (default: 55)')
parser.add_argument('--job_size_pareto_shape', type=float, default=1.6, help='pareto job size distribution shape (default: 1.5)')
parser.add_argument('--job_size_pareto_scale', type=float, default=1, help='pareto job size distribution scale (default: 100.0)')
parser.add_argument('--load_balance_queue_size', type=int, default=100, help='observation cap for load balance env (default: 500000.0)')
#parser.add_argument('--load_balance_queue_size', type=int, default=6, help='observation cap for load balance env (default: 500000.0)')

config, _ = parser.parse_known_args()