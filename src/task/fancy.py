from collections import OrderedDict

from task.square import extract_xythetas, extract_xys
from . import Trace
import numpy as np
import matplotlib as mpl

mpl.rcParams['savefig.pad_inches'] = 0
from matplotlib import pyplot as plt
plt.rcParams["animation.html"] = "jshtml"
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation
from matplotlib import rc


def plot_task(ti_mat, tt, L=1, title=''):
    n_robots = len(ti_mat)

    plt.figure(figsize=(6, 6))
    ax = plt.gcf().gca()

    ax.set_xlim([-0.5, L + 0.5])
    ax.set_ylim([-0.5, L + 0.5])
    ax.axis("equal")
    ax.grid(True)

    ti, angles = extract_xythetas(ti_mat)

    plt.plot(ti[:, 0], ti[:, 1], 'bo', ms=6, c='b', label='robots')
    plt.plot(tt[:, 0], tt[:, 1], 'X', ms=6, c='r', label='targets')

    hattoro = np.array([[0, 0, 1], [0.1, 0, 1]]).T
    for f in ti_mat:
        xhat = f @ hattoro
        ax.plot(xhat[0, :], xhat[1, :], 'k-.')

    intent = [ax.add_line(
        Line2D([ti[j, 0], tt[j, 0]], [ti[j, 1], tt[j, 1]], linestyle='--', color='black', linewidth=1.5, alpha=0.8))
        for j in range(n_robots)]

    # plt.axis('off')
    plt.title(title)
    plt.legend()
    plt.show()


def plot_trace(trace: Trace, task, L: float = 1, show=False, save_to='') -> None:

    if len(trace.communication[0]) > 0:
        n = 4
    else:
        n = 3
    fig, axs = plt.subplots(1, n, figsize=(16, 5))
    tracestate = np.array([extract_xys(s) for s in trace.pos_state])
    axs[0].plot(tracestate[:, :, 0], tracestate[:, :, 1], '.-')
    axs[0].plot(tracestate[-1, :, 0], tracestate[-1, :, 1], 'bo', c='b', ms=7)
    axs[0].plot(task.targets[:, 0], task.targets[:, 1], 'X', c='r', ms=7)
    axs[0].set_ylabel('time')
    axs[0].set_title('position')
    axs[0].axis("equal")
    axs[0].grid(True)
    axs[0].set_xlim((-0.5, L + 0.5))
    axs[0].set_ylim((-0.5, L + 0.5))

    axs[1].plot(trace.control[:, :, 0], trace.time, '.-', color='r', label='linear(x)')
    axs[1].set_ylabel('time')
    axs[1].set_title('speed')
    axs1b = axs[1].twiny()
    axs1b.plot(trace.control[:, :, 1], trace.time, '.-', color='b', label='angular')

    handles, labels = axs[1].get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    axs[1].legend(by_label.values(), by_label.keys())

    handles, labels = axs1b.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    axs1b.legend(by_label.values(), by_label.keys(), loc = 'lower right')

    axs[2].plot(trace.error, trace.time, 'k.-')
    axs[2].set_ylabel('time')
    axs[2].set_title('error')

    if len(trace.communication[0]) > 0:
        axs[-1].plot(trace.communication[:, 0, :], trace.time, '.-')
        axs[-1].set_xlabel('time')
        axs[-1].set_title('communication')

    if save_to != '':
        plt.savefig(save_to)
    if show:
        plt.show()


def animate_with_targets(trace: Trace, sensor, show=False, save_to: str = '', dt: int = 0.1, L: float = 1):
    dev_room = [L * 3, L * 3]

    fig = plt.figure(figsize=(7, 7))

    extra = 0.2
    ax = fig.add_subplot(aspect='equal', autoscale_on=True,
                         xlim=np.array([0, L]) + [-extra, +extra],
                         ylim=np.array([0, L]) + [-extra, +extra])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.axis('off')

    plt.autoscale(tight=True)

    robots, = ax.plot([], [], 'bo', ms=9, c='b', zorder=4)
    targets, = ax.plot([], [], 'X', ms=9, c='r', zorder=3)

    n_robots = trace.pos_state.shape[1]

    intent = [ax.add_line(Line2D([], [], linestyle='--', color='black', linewidth=3, alpha=0.8))
              for _ in range(n_robots)]

    orientation = [ax.add_line(Line2D([], [], linestyle='-.', color='black', linewidth=3, zorder=5))
              for _ in range(n_robots)]

    robotext = [ax.text(dev_room[0], dev_room[1], i,
                        ha="center", fontweight='bold', fontsize=20, zorder=2) for i in range(n_robots)]

    targtext = [ax.text(dev_room[0], dev_room[1], i,
                        ha="center", fontweight='bold', fontsize=20, zorder=2) for i in range(n_robots)]

    disttext = [ax.text(dev_room[0], dev_room[1], '', ha="center", fontsize=20, zorder=1)
                for _ in range(n_robots)]

    commtext = [ax.text(dev_room[0], dev_room[1], '', color='black', ha='center', va='center',
                        bbox=dict(facecolor='none', edgecolor='blue'))
                for i in range(n_robots)]


    if "range" in sensor.__qualname__:
        range_radius = sensor.get_params()
    else:
        range_radius = L * 6

    circles = [ax.add_artist(plt.Circle(dev_room, range_radius, color='b', linestyle='dotted',
                                        alpha=0.5, fill=False)) for r in range(n_robots)]

    def init():
        robots.set_data([], [])
        targets.set_data([], [])
        for i in range(n_robots):
            intent[i].set_data([], [])
            orientation[i].set_data([], [])

        return (robots, targets, *intent, *robotext, *targtext, *disttext, *circles, *commtext, *orientation)

    def update(i):
        pos_state = trace.pos_state[i]
        optimal_state = trace.targets[i]

        xys, thetas = extract_xythetas(pos_state)
        robots.set_data(xys[:, 0], xys[:, 1])

        targets.set_data(optimal_state[:, 0], optimal_state[:, 1])

        robot_face = np.array([[0, 0, 1], [0.1, 0, 1]]).T
        for j, (r_mat, r_xy, r_angle, r_target) in enumerate(zip(pos_state, xys, thetas, optimal_state)):

            xhat = r_mat @ robot_face
            orientation[j].set_data(xhat[0, :], xhat[1, :])

            rx, ry = r_xy
            tx, ty = r_target

            circles[j].center = (rx, ry)

            angle = np.arctan2(ty - ry, tx - rx)
            offset_x, offset_y = 0.075 * np.cos(angle - np.pi), 0.075 * np.sin(angle - np.pi)

            if len(trace.communication[i]) > 0:
                txt = trace.communication[i, :2, j]
                commtext[j].set_text(f"{txt[0]:.2f}\n{txt[0]:.2f}")
                offset_x, offset_y = 0.12 * np.cos(angle + np.pi), 0.12 * np.sin(angle + np.pi)
                commtext[j].set_position((rx + offset_x, ry + offset_y))

            intent[j].set_data((rx, tx), (ry, ty))

            dist = np.sqrt((rx - tx) ** 2 + (ry - ty) ** 2)
            if dist > 0.25:
                targtext[j].set_position((tx - offset_x, ty - offset_y))

                extra = 0.22
                if len(trace.communication[i]) == 0:
                    extra = 0.10

                offset2_x, offset2_y = extra * np.cos(angle + np.pi), extra * np.sin(angle + np.pi)
                robotext[j].set_position((rx + offset2_x, ry + offset2_x))
                disttext[j].set_position((np.mean([rx, tx]), np.mean([ry, ty])))
                disttext[j].set_text("{:.2f}".format(dist))
            else:
                if dist > 0.08:
                    targtext[j].set_position((tx - offset_x, ty - offset_y))
                else:
                    targtext[j].set_position((tx - 0.1, ty - 0.02))
                robotext[j].set_position(dev_room)
                disttext[j].set_position((dev_room, dev_room))
                disttext[j].set_text("")

                # commtext[j].set_text("")
                # commtext[j].set_position(dev_room)

        return (robots, targets, *intent, *robotext, *targtext, *disttext, *circles, *commtext, *orientation)

    ani = FuncAnimation(fig, update, frames=len(trace.pos_state),
                        init_func=init, blit=True, interval=1000 * dt)

    if save_to:
        ani.save(save_to)
    if show:
        plt.show()
    plt.close(fig)

    rc('animation', html='jshtml')
    return ani
