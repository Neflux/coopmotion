from task.square import RADIUS
from . import Trace
import numpy as np
import matplotlib as mpl
mpl.rcParams['savefig.pad_inches'] = 0


def plot_task(ti, tt, L=1, title=''):
    from matplotlib import pyplot as plt
    n_robots = len(ti)

    plt.figure(figsize=(6, 6))
    ax = plt.gcf().gca()

    ax.set_xlim([-0.5, L + 0.5])
    ax.set_ylim([-0.5, L + 0.5])
    ax.axis("equal")
    ax.grid(True)

    plt.plot(ti[:, 0], ti[:, 1], 'bo', ms=6, c='b', label='robots')
    plt.plot(tt[:, 0], tt[:, 1], 'X', ms=6, c='r', label='targets')
    from matplotlib.lines import Line2D
    intent = [ax.add_line(
        Line2D([ti[j, 0], tt[j, 0]], [ti[j, 1], tt[j, 1]], linestyle='--', color='black', linewidth=1.5, alpha=0.8))
        for j in range(n_robots)]

    # plt.axis('off')
    plt.title(title)
    plt.legend()
    plt.show()


def plot_trace(trace: Trace, L: float = 1, show=False, save_to = '') -> None:
    from matplotlib import pyplot as plt
    if len(trace.communication[0]) > 0:
        n = 4
    else:
        n = 3
    fig, axs = plt.subplots(1, n, figsize=(16, 5))
    axs[0].plot(trace.state[:, :, 0], trace.state[:, :, 1], '.-')
    axs[0].set_ylabel('time')
    axs[0].set_title('position')
    axs[0].axis("equal")
    axs[0].grid(True)
    axs[0].set_xlim((-0.5, L + 0.5))
    axs[0].set_ylim((-0.5, L + 0.5))
    axs[1].plot(trace.control[:, :, 0], trace.time, '.-', color='r')
    axs[1].set_ylabel('time')
    axs[1].set_title('speed')
    axs[1].plot(trace.control[:, :, 1], trace.time, '.-', color='b')
    axs[2].plot(trace.error, trace.time, 'k.-')
    axs[2].set_xlabel('time')
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
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.animation import FuncAnimation
    from matplotlib import rc

    dev_room = [L * 3, L * 3]

    rc('animation', html='html5')
    fig = plt.figure(figsize=(7,7))

    extra = 0.2
    ax = fig.add_subplot(aspect='equal', autoscale_on=True, frameon=False,
                         xlim=np.array([0, L]) + [-extra, +extra],
                         ylim=np.array([0, L]) + [-extra, +extra])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.axis('off')

    plt.autoscale(tight=True)

    robots, = ax.plot([], [], 'bo', ms=9, c='b', zorder=4)
    targets, = ax.plot([], [], 'X', ms=9, c='r', zorder=3)

    n_robots = trace.state.shape[1]

    intent = [ax.add_line(Line2D([], [], linestyle='--', color='black', linewidth=3, alpha=0.8))
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

        return (robots, targets, *intent, *robotext, *targtext, *disttext, *circles, *commtext)

    def update(i):
        frame, optimal_frame = trace.state[i], trace.targets[i]
        rx, ry = frame[:, 0], frame[:, 1]
        robots.set_data(rx, ry)

        tx, ty = optimal_frame[:, 0], optimal_frame[:, 1]
        targets.set_data(tx, ty)

        for j in range(n_robots):
            circles[j].center = (rx[j], ry[j])

            #angle = np.arctan2(ty[j] - ty.mean(), tx[j] - tx.mean())
            #offset_x, offset_y = 0.075 * np.cos(angle-np.pi), 0.075 * np.sin(angle-np.pi)
            #targtext[j].set_position((tx[j] - offset_x, ty[j] - offset_y - 0.02))
            angle = np.arctan2(ty[j] - ry[j], tx[j] - rx[j])
            offset_x, offset_y = 0.075 * np.cos(angle-np.pi), 0.075 * np.sin(angle-np.pi)


            if len(trace.communication[i]) > 0:
                txt = trace.communication[i,:2,j]
                commtext[j].set_text(f"{txt[0]:.2f}\n{txt[0]:.2f}")
                offset_x, offset_y = 0.12 * np.cos(angle + np.pi), 0.12 * np.sin(angle + np.pi)
                commtext[j].set_position((rx[j]+offset_x, ry[j]+offset_y))

            intent[j].set_data((rx[j], tx[j]), (ry[j], ty[j]))

            dist = np.sqrt((rx[j] - tx[j]) ** 2 + (ry[j] - ty[j]) ** 2)
            if dist > 0.25:
                targtext[j].set_position((tx[j] - offset_x, ty[j] - offset_y))
                #angle = np.arctan2(ry[j] - ry.mean(), rx[j] - rx.mean())
                extra = 0.22
                if len(trace.communication[i]) == 0:
                    extra = 0.10
                offset2_x, offset2_y = extra * np.cos(angle+np.pi), extra * np.sin(angle+np.pi)
                robotext[j].set_position((rx[j] + offset2_x, ry[j] + offset2_x))
                disttext[j].set_position((np.mean([rx[j], tx[j]]), np.mean([ry[j], ty[j]])))
                disttext[j].set_text("{:.2f}".format(dist))
            else:
                if dist > 0.08:
                    targtext[j].set_position((tx[j] - offset_x, ty[j] - offset_y))
                else:
                    targtext[j].set_position((tx[j] - 0.1, ty[j] - 0.02))
                robotext[j].set_position(dev_room)
                disttext[j].set_position((dev_room, dev_room))
                disttext[j].set_text("")

                #commtext[j].set_text("")
                #commtext[j].set_position(dev_room)

        return (robots, targets, *intent, *robotext, *targtext, *disttext, *circles, *commtext)

    ani = FuncAnimation(fig, update, frames=len(trace.state),
                        init_func=init, blit=True, interval=1000 * dt)

    if save_to:
        ani.save(save_to)
    if show:
        plt.show()
    plt.close(fig)
    return ani
