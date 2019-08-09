import os
import pathlib
import pickle
import matplotlib.pyplot as plt

# colab
import sys

sys.path.append('coopmotion/src')

from network import train_net, CentralizedNet, EnhancedDistributedNet
from com_network import ComNet
from task.dataset import *
from task.dataset import generate_sequential_dataset
from task.fancy import animate_with_targets
from task.square import *
from task.square import dynamic, sense_all_fixed


# dir_path = os.path.dirname(os.path.realpath(__file__))

# For any dataset
def save_stuff(file_name, folder, stuff):
    if 'google.colab' in sys.modules:
        folder = os.path.join('coopmotion', folder)
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(folder, f'{file_name}.pkl'), 'wb') as f:
        pickle.dump(stuff, f)


def load_stuff(file_name, folder):
    if 'google.colab' in sys.modules:
        folder = os.path.join('coopmotion', folder)
    with open(os.path.join(folder, f'{file_name}.pkl'), 'rb') as f:
        tmp = pickle.load(f)
    return tmp


# For any pytorch network
def save(net, net_name, task, N, controller, sensor, samples, folder='models'):
    if 'google.colab' in sys.modules:
        folder = os.path.join('coopmotion', folder)
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
    torch.save(net, os.path.join(folder, f"{net_name}_{craft_name(task, N, controller, sensor, samples)}.model"))


def load(net_name, task, N, controller, sensor, samples, folder='models'):
    if 'google.colab' in sys.modules:
        folder = os.path.join('coopmotion', folder)
    file_name = os.path.join(folder, f"{net_name}_{craft_name(task, N, controller, sensor, samples)}.model")
    if os.path.exists(file_name):
        return torch.load(file_name)
    else:
        return None


def craft_name(task, N, controller, sensor, samples):
    sensor_string = sensor.__qualname__[:sensor.__qualname__.find('.')]
    return f'task={type(task).__name__}_N={N}_controller={controller.__name__}_sensor={sensor_string}{sensor.get_params()}_samples={samples}'


# Plotting
def plot_error(time, error, show=False, **kwargs):
    m = np.mean(error, axis=0)
    q95 = np.quantile(error, 0.95, axis=0)
    q5 = np.quantile(error, 0.05, axis=0)
    label = kwargs.pop('label', '')
    plt.plot(time, m, '-', label=label, **kwargs)
    plt.fill_between(time, q5, q95, alpha=0.1, **kwargs)
    if show:
        plt.title('Error')
        plt.legend()
        plt.savefig('error')
        plt.show()


def visualize_network_behaviour(net_run, T=10, show=False, save_to=''):
    trace = net_run(T=T, epsilon=EPS)
    #plot_trace(trace)
    animate_with_targets(trace, net_run.sensor, show=show, save_to=save_to)


# Constant parameters
COMPARISON_SIMS = 100
controller = proportional
N = 4
EPS = 0.001
number_of_samples = 100
""" Centralized """

# general settings
task = static_evenly_spread_task(N)
dataset_sensing_mode = sense_all_fixed()  # It's ignored anyway
run = SegmentRun(task, controller=controller, sensor=dataset_sensing_mode, dt=0.1)

# load if possible, otherwise cache the expert dataset
sensor_string = dataset_sensing_mode.__qualname__[:dataset_sensing_mode.__qualname__.find('.')]
file_base = craft_name(task, N, controller, dataset_sensing_mode, number_of_samples)
file_name = os.path.join("dataset/centralized", f'{file_base}.pkl')
if 'google.colab' in sys.modules:
    file_name = os.path.join('coopmotion', file_name)
redo = False
if os.path.isfile(file_name):
    print(f"[Centralized] Cached dataset was loaded from {file_base}")
    train_dataset, test_dataset, exact_traces, exact_error = load_stuff(file_base, 'dataset/centralized')
else:
    print(f"[Centralized] Creating new dataset in {file_base}")

    # For training and testing
    train_dataset = central_dataset(generate_non_sequential_dataset(run=run, number=number_of_samples, epsilon=EPS))
    test_dataset = central_dataset(generate_non_sequential_dataset(run=run, number=number_of_samples, epsilon=EPS))

    # For actual evaluation
    exact_traces = generate_sequential_dataset(run, number=COMPARISON_SIMS, duration=10, epsilon=EPS)
    exact_traces = [prepare(t, padding=True, steps=50) for t in exact_traces]
    exact_error = np.array([t.error for t in exact_traces])
    save_stuff(file_base, "dataset/centralized", [train_dataset, test_dataset, exact_traces, exact_error])
    redo = True
time = exact_traces[0].time

# load if possible, otherwise train the net
net = load('centralized', task, N, controller, dataset_sensing_mode, number_of_samples)
if net is None or redo:
    net = CentralizedNet(N)
    train_net(epochs=500, net=net, train_dataset=train_dataset, test_dataset=test_dataset, batch_size=100)
    save(net, 'centralized', task, N, controller, dataset_sensing_mode, number_of_samples)
    redo = True
else:
    print("[Centralized] Network weights loaded from file")

net_controller = net.controller()
net_run = Run(task, dynamic=dynamic(dt=0.1), sensor=dataset_sensing_mode, dt=0.1, controller=net_controller)

file_base = f"centsim_{craft_name(task, N, controller, dataset_sensing_mode, number_of_samples)}"
file_name = os.path.join("dataset/centralized", f'{file_base}.pkl')
if 'google.colab' in sys.modules:
    file_name = os.path.join('coopmotion', file_name)
if os.path.isfile(file_name) and not redo:
    print(f'[Centralized] Cached simulations loaded successfully')
    net_error = load_stuff(file_base, 'dataset/centralized')
else:
    print(f'[Centralized] Running {COMPARISON_SIMS} simulations')
    net_traces = generate_sequential_dataset(net_run, number=COMPARISON_SIMS, duration=10, epsilon=EPS)
    net_traces = [prepare(t, padding=True, steps=50) for t in net_traces]
    net_error = np.array([t.error for t in net_traces])
    save_stuff(file_base, "dataset/centralized", net_error)

plot_error(time, net_error, color='grey', label='centralized')

""" Distributed """

# general settings
number_of_samples = 100
task = dynamic_evenly_spread_task(N)
sensor_testing_pipeline = [
    sense_in_range(0.7)
]

from matplotlib import cm

colors = [cm.Dark2(x) for x in range(len(sensor_testing_pipeline))]

for i, dataset_sensing_mode in enumerate(sensor_testing_pipeline):
    # load if possible, otherwise cache the expert dataset
    sensor_string = dataset_sensing_mode.__qualname__[:dataset_sensing_mode.__qualname__.find('.')]
    sensor_string_clean = sensor_string.replace('_', ' ') + ' (' + str(dataset_sensing_mode.get_params()) + ')'
    file_base = craft_name(task, N, controller, dataset_sensing_mode, number_of_samples)
    file_name = os.path.join("dataset/distributed", f'{file_base}.pkl')
    if 'google.colab' in sys.modules:
        file_name = os.path.join('coopmotion', file_name)
    redo = False
    if os.path.isfile(file_name):
        print(f"[Distributed - {sensor_string_clean}] Cached dataset was loaded from {file_base}")
        d_train_dataset, d_test_dataset = load_stuff(file_base, 'dataset/distributed')
    else:
        print(f"[Distributed - {sensor_string_clean}] Creating new dataset in {file_base}")
        run = SegmentRun(task, controller=controller, sensor=dataset_sensing_mode, dt=0.1)
        d_train_dataset = distributed_dataset(generate_non_sequential_dataset(
            run=run, number=number_of_samples, epsilon=EPS))
        d_test_dataset = distributed_dataset(generate_non_sequential_dataset(
            run=run, number=number_of_samples, epsilon=EPS))
        save_stuff(file_base, "dataset/distributed", [d_train_dataset, d_test_dataset])
        redo = True

    # load if possible, otherwise train the net
    net = load('distributed', task, N, controller, dataset_sensing_mode, number_of_samples)
    if net is None or redo:
        net = EnhancedDistributedNet(dataset_sensing_mode.get_input_size(N))
        train_net(epochs=500, net=net, train_dataset=d_train_dataset, test_dataset=d_test_dataset, batch_size=100)
        save(net, 'distributed', task, N, controller, dataset_sensing_mode, number_of_samples)
        redo = True
    else:
        print(f"[Distributed - {sensor_string_clean}] Network weights loaded from file")

    net_controller = net.controller()
    net_run = Run(task, dynamic=dynamic(dt=0.1), sensor=dataset_sensing_mode, dt=0.1, controller=net_controller)

    file_base = f"distsim_{craft_name(task, N, controller, dataset_sensing_mode, number_of_samples)}"
    file_name = os.path.join("dataset/distributed", f'{file_base}.pkl')
    if 'google.colab' in sys.modules:
        file_name = os.path.join('coopmotion', file_name)
    if os.path.isfile(file_name) and not redo:
        print(f'[Distributed - {sensor_string_clean}] Cached simulations loaded successfully')
        net_error = load_stuff(file_base, 'dataset/distributed')
    else:
        print(f'[Distributed - {sensor_string_clean}] Running {COMPARISON_SIMS} simulations')
        net_traces = generate_sequential_dataset(net_run, number=COMPARISON_SIMS, duration=10, epsilon=0.01)
        net_traces = [prepare(t, padding=True, steps=50) for t in net_traces]
        net_error = np.array([t.error for t in net_traces])
        save_stuff(file_base, "dataset/distributed", net_error)

    plot_error(time, net_error, color=colors[i], label=sensor_string_clean)
    #visualize_network_behaviour(net_run, show=True)

plot_error(time, exact_error, show=True, color='black', label='exact')

""" Distributed with communication """

# general settings
UNROLL_WINDOW = 2
number_of_samples = 100
epochs: int = 500
batch_size = 100
broadcast_size = 2
task = dynamic_evenly_spread_task(N)
sensor_testing_pipeline = [
    sense_in_range(0.7)
]

from matplotlib import cm

colors = [cm.Set1(x) for x in range(len(sensor_testing_pipeline))]

for i, dataset_sensing_mode in enumerate(sensor_testing_pipeline):
    # load if possible, otherwise cache the expert dataset
    sensor_string = dataset_sensing_mode.__qualname__[:dataset_sensing_mode.__qualname__.find('.')]
    sensor_string_clean = f"{sensor_string.replace('_', ' ')} comm ({dataset_sensing_mode.get_params()})"
    file_base = craft_name(task, N, controller, dataset_sensing_mode, number_of_samples)+f"_unrollwin={UNROLL_WINDOW}"
    file_name = os.path.join("dataset/distributed_comm", f'{file_base}.pkl')
    if 'google.colab' in sys.modules:
        file_name = os.path.join('coopmotion', file_name)
    redo = False
    if os.path.isfile(file_name):
        print(f"[Distributed - {sensor_string_clean}] Cached dataset was loaded from {file_base}")
        c_train_dataset, c_test_dataset = load_stuff(file_base, 'dataset/distributed_comm')
    else:
        print(f"[Distributed - {sensor_string_clean}] Creating new dataset in {file_base}")
        run = SegmentRun(task, controller=controller, sensor=dataset_sensing_mode, dt=0.1)
        train_traces = generate_sequential_dataset(run=run, number=number_of_samples)
        test_traces = generate_sequential_dataset(run=run, number=number_of_samples)
        c_train_dataset = sequence_dataset(train_traces, steps=UNROLL_WINDOW)
        c_test_dataset = sequence_dataset(test_traces, steps=UNROLL_WINDOW)

        save_stuff(file_base, "dataset/distributed_comm", [c_train_dataset, c_test_dataset])
        redo = True

    # load if possible, otherwise train the net
    net = load('distributed_comm', task, N, controller, dataset_sensing_mode, number_of_samples)
    if net is None or redo:
        net = ComNet(N=N, broadcast=broadcast_size, batch_size=batch_size, unroll_window=UNROLL_WINDOW)
        train_net(epochs=epochs, net=net, train_dataset=c_train_dataset, test_dataset=c_test_dataset, batch_size=batch_size)
        save(net, 'distributed_comm', task, N, controller, dataset_sensing_mode, number_of_samples)
        redo = True
    else:
        print(f"[Distributed - {sensor_string_clean}] Network weights loaded from file")

    net_controller = net.controller()
    net_run = Run(task, dynamic=dynamic(dt=0.1), sensor=dataset_sensing_mode, dt=0.1, controller=net_controller)

    file_base = f"distcommsim_{craft_name(task, N, controller, dataset_sensing_mode, number_of_samples)}"
    file_name = os.path.join("dataset/distributed_comm", f'{file_base}.pkl')
    if 'google.colab' in sys.modules:
        file_name = os.path.join('coopmotion', file_name)
    if os.path.isfile(file_name) and not redo:
        print(f'[Distributed - {sensor_string_clean}] Cached simulations loaded successfully')
        net_error = load_stuff(file_base, 'dataset/distributed_comm')
    else:
        print(f'[Distributed - {sensor_string_clean}] Running {COMPARISON_SIMS} simulations')
        net_traces = generate_sequential_dataset(net_run, number=COMPARISON_SIMS, duration=10, epsilon=0.01)
        net_traces = [prepare(t, padding=True, steps=50) for t in net_traces]
        net_error = np.array([t.error for t in net_traces])
        save_stuff(file_base, "dataset/distributed_comm", net_error)

    plot_error(time, net_error, color=colors[i], label=sensor_string_clean)

plot_error(time, exact_error, show=True, color='black', label='exact')

#visualize_network_behaviour(net_run, show=True, save_to='comm.mp4')
