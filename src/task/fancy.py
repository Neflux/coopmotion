import os
import pathlib
from collections import OrderedDict

from task.math import extract_tuple_from_state, extract_xys_from_state, is_homogenous
from . import Trace
import numpy as np
import matplotlib as mpl

mpl.rcParams['savefig.pad_inches'] = 0
from matplotlib import pyplot as plt

plt.rcParams["animation.html"] = "jshtml"
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation
from matplotlib import rc

from utils import detailed_name


def plot_task(ti, tt, L=1, title='', show=True):
    if not show:
        return

    n_robots = len(ti)

    plt.figure(figsize=(6, 6))
    ax = plt.gcf().gca()

    ax.set_xlim([-0.5, L + 0.5])
    ax.set_ylim([-0.5, L + 0.5])
    ax.axis("equal")
    ax.grid(True)


    poses = None
    if is_homogenous(ti):
        poses = ti
        ti = extract_xys_from_state(ti)


    plt.plot(ti[:, 0], ti[:, 1], 'bo', ms=6, c='b', label='robots', zorder=2)
    plt.plot(tt[:, 0], tt[:, 1], 'X', ms=8, c='r', label='targets', zorder=2)

    for j in range(n_robots):
        ax.add_line(Line2D([ti[j, 0], tt[j, 0]], [ti[j, 1], tt[j, 1]],
                           linestyle='--', color='black', linewidth=1.5, alpha=0.8, zorder=3))

    if poses is not None:
        x_axis = np.array([[0, 0, 1], [0.08, 0, 1]]).T
        y_axis = np.array([[0, 0, 1], [0, 0.04, 1]]).T
        for f in poses:
            xhat = f @ x_axis
            yhat = f @ y_axis
            ax.plot(xhat[0, :], xhat[1, :], 'r-')
            ax.plot(yhat[0, :], yhat[1, :], 'g-')

    plt.title(title)
    plt.legend()
    plt.show()


def components(a, b):
    return [a[0], a[0]], [a[1], b[1]], [b[0], a[0]], [b[1],b[1]]


def plot_sensing(state, sense, L=1):
    plt.figure(figsize=(6, 6))
    ax = plt.gcf().gca()

    ax.set_xlim([-2.5, L + 2.5])
    ax.set_ylim([-2.5, L + 2.5])
    ax.axis("equal")
    ax.grid(True)

    xys = state
    if is_homogenous(state):
        xys = extract_xys_from_state(state)

    plt.plot(xys[:, 0], xys[:, 1], 'bo', ms=6, c='b', label='robots')

    sensing = sense(state)

    if is_homogenous(state):
        if sense.range is None:
            sensing = np.dstack([sensing, np.ones((len(state), sensing.shape[1]))])

        x_axis = np.array([[0, 0, 1], [0.1, 0, 1]]).T
        y_axis = np.array([[0, 0, 1], [0, 0.1, 1]]).T
        for i, f in enumerate(state):
            xhat = f @ x_axis
            yhat = f @ y_axis
            ax.plot(xhat[0, :], xhat[1, :], 'r-')
            ax.plot(yhat[0, :], yhat[1, :], 'g-')
            if sense.range is not None:
                ax.add_artist(plt.Circle(xys[i],sense.range, color='b', linestyle='dotted', alpha=0.7, fill=False))
            ax.text(f[0, 2] + 0.05, f[1, 2], str(i), zorder=5, fontsize=18)

            for j, rel_neighbor in enumerate(sensing[i]):
                abs_neighbor = (rel_neighbor @ np.linalg.inv(f)) + f[:, 2]
                ax.add_line(Line2D([xys[i, 0], abs_neighbor[0]], [xys[i, 1], abs_neighbor[1]],
                                   linestyle='--', color='grey', linewidth=1.0, alpha=0.7))
    else:
        plt.plot(xys[0, 0], xys[0, 1], 'bo', ms=7, c='black')
        for i, rel_neighbor in enumerate(sensing[0]):

            abs_neighbor = rel_neighbor[:2] + xys[0]
            a,b,c,d = components(xys[0], abs_neighbor)
            ax.add_line(Line2D(a,b, linestyle='--', color='r', linewidth=1.0, alpha=0.9))
            ax.add_line(Line2D(c,d, linestyle='--', color='g', linewidth=1.0, alpha=0.9))

    plt.title('Sensing')
    plt.legend()
    plt.show()


def training_plot(history, context=None):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Plot training & validation loss values
    axs[0].plot(history.history['loss'])
    axs[0].plot(history.history['val_loss'])
    axs[0].set_title('Model loss')
    axs[0].set_ylabel('Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['Training', 'Validation'], loc='upper right')

    # Plot training & validation accuracy values
    axs[1].plot(history.history['acc'])
    axs[1].plot(history.history['val_acc'])
    axs[1].set_title('Model accuracy')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['Training', 'Validation'], loc='upper right')

    plt.show()

    if context is not None:
        folder = os.path.join('images', 'training')
        path = os.path.join(folder, detailed_name(context) + '.png')
        print(f"Saved in {path}")
        pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
        plt.savefig(path)


def plot_trace(trace: Trace, task, L: float = 1, show=False, context=None):
    if len(trace.communication[0]) > 0:
        n = 4
    else:
        n = 3
    fig, axs = plt.subplots(1, n, figsize=(16, 5))
    tracestate = trace.state
    if is_homogenous(trace.state[0]):
        tracestate = np.array([extract_xys_from_state(s) for s in trace.state])
    axs[0].plot(tracestate[:, :, 0], tracestate[:, :, 1], '.-')
    axs[0].plot(tracestate[-1, :, 0], tracestate[-1, :, 1], 'bo', c='b', ms=7)
    axs[0].plot(task.target_xys[:, 0], task.target_xys[:, 1], 'X', c='r', ms=7)
    axs[0].set_ylabel('time')
    axs[0].set_title('position')
    axs[0].axis("equal")
    axs[0].grid(True)
    axs[0].set_xlim((-0.5, L + 0.5))
    axs[0].set_ylim((-0.5, L + 0.5))

    tracecontrol = trace.control[:, :, 0]
    axs[1].set_ylabel('time')
    axs[1].set_title('speed')
    if is_homogenous(trace.state[0]):
        #TODO: 2nd y axis inside the plot, so the titles align
        axs[1].plot(np.flip(trace.control[:, :, 0],0), trace.time, '.-', color='r', label='linear(x)')
        axs1b = axs[1].twiny()
        axs1b.plot(trace.control[:, :, 1], trace.time, '.-', color='b', label='angular')

        handles, labels = axs[1].get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        axs[1].legend(by_label.values(), by_label.keys())

        handles, labels = axs1b.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        axs1b.legend(by_label.values(), by_label.keys(), loc='lower right')
    else:
        axs[1].plot(trace.control[:, :, 0], trace.time, '.-', color='r')

    axs[2].plot(trace.error, trace.time, 'k.-')
    axs[2].set_ylabel('time')
    axs[2].set_title('error')

    if len(trace.communication[0]) > 0:
        axs[-1].plot(trace.communication[:, 0, :], trace.time, '.-')
        axs[-1].set_xlabel('time')
        axs[-1].set_title('communication')

    if context is not None:
        folder = os.path.join('images', 'trace')
        path = os.path.join(folder, detailed_name(context) + '.png')
        print(f"Saved in {path}")
        pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
        plt.savefig(path)
    if show:
        plt.show()


def animate_with_targets(trace: Trace, sensor, show=False, context=None, dt: int = 0.1, L: float = 1):
    dev_room = [L * 3, L * 3]

    fig = plt.figure(figsize=(7, 7))

    extra = 0.5
    ax = fig.add_subplot(aspect='equal', autoscale_on=True,
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


    def h_update(i):
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

        return (robots, targets, *intent, *robotext, *targtext, *disttext, *circles, *commtext, *orientation)


    def nh_update(i):
        state = trace.state[i]
        optimal_state = trace.targets[i]

        xys, thetas = extract_tuple_from_state(state)
        robots.set_data(xys[:, 0], xys[:, 1])

        targets.set_data(optimal_state[:, 0], optimal_state[:, 1])

        robot_face = np.array([[0, 0, 1], [0.1, 0, 1]]).T
        for j, (r_mat, r_xy, r_angle, r_target) in enumerate(zip(state, xys, thetas, optimal_state)):

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
    print((trace.state.shape))
    ani = FuncAnimation(fig, nh_update if is_homogenous(trace.state[0]) else h_update, frames=len(trace.state),
                        init_func=init, blit=True, interval=1000 * dt)

    if context is not None:
        folder = os.path.join('videos')
        path = os.path.join(folder, detailed_name(context) + '.mp4')
        print(f"Saved in {path}")
        pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
        ani.save(path)
    if show:
        plt.show()
    plt.close(fig)

    rc('animation', html='jshtml')
    return ani


def plot_error(time, error, context=None, **kwargs):
    m = np.mean(error, axis=0)
    q95 = np.quantile(error, 0.95, axis=0)
    q5 = np.quantile(error, 0.05, axis=0)
    label = kwargs.pop('label', '')
    plt.plot(time, m, '-', label=label, **kwargs)
    plt.fill_between(time, q5, q95, alpha=0.1, **kwargs)

    if context is not None:
        plt.legend()
        plt.savefig('centralized')
        plt.title('Error')

        folder = os.path.join('images', 'error')
        path = os.path.join(folder, detailed_name(context) + '.png')
        print(f"Saved in {path}")
        pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
        plt.savefig(path)

def plot_dataset(t, r):
    fig, axs = plt.subplots(2,2, figsize=(16,10))
    if r.task.holonomic:
        axs[0,0].set_title('Position (x)')
        for state in np.moveaxis(t.state,0,-1):
            state = np.moveaxis(state,-1,0)
            axs[0,0].hist(state[:,0], bins=30, alpha=0.5);
        axs[0,1].set_title('Sensing (first robot dx, fixed or sorted)')
        for sensing in np.moveaxis(t.sensing,0,-1):
            sensing = np.moveaxis(sensing,-1,0)
            axs[0,1].hist(sensing[:,0,0], bins=30, alpha=0.5);
        axs[1,0].set_title('Control (vx)')
        for control in np.moveaxis(t.control,0,-1):
            control = np.moveaxis(control,-1,0)
            axs[1,0].hist(control[:,0], bins=30, alpha=0.5);
        axs[1,1].set_title('Error')
        axs[1,1].hist(t.error, bins=60, alpha=0.5);
        #axs[1,1].set_title('Targets')
        #for targets in np.moveaxis(t.targets,0,-1):
        #    targets = np.moveaxis(targets,-1,0)
        #    axs[1,1].hist(targets[:,0], bins=30, alpha=0.5);
    else:
        axs[0,0].set_title('Position (x)')
        for state in np.moveaxis(t.state,0,-1):
            state = extract_xys_from_state(np.moveaxis(state,-1,0))
            axs[0,0].hist(state[:,0], bins=30, alpha=0.5);
        axs[0,1].set_title('Sensing (first robot dx, fixed or sorted)')
        for sensing in np.moveaxis(t.sensing,0,-1):
            sensing = np.moveaxis(sensing,-1,0)
            axs[0,1].hist(sensing[:,0,0], bins=30, alpha=0.5);
        axs[1,0].set_title('Control (linear velocity)')
        for control in np.moveaxis(t.control,0,-1):
            control = np.moveaxis(control,-1,0)
            axs[1,0].hist(control[:,0], bins=30, alpha=0.5);
        axs[1,1].set_title('Error')
        axs[1,1].hist(t.error, bins=60, alpha=0.5);


def hist3d(trace, n_robots, bins, height, title):
    import plotly.graph_objects as go
    import colorlover as cl
    ryb = cl.scales['11']['qual']['Paired']

    fig = go.Figure()
    fig.update_layout(title=go.layout.Title(
        text=title,
        xref="paper",
        x=0
    ), scene = dict(
                        xaxis = dict(nticks=4, range=[0,1],),
                        yaxis = dict(nticks=4, range=[0,1],),
                             zaxis = dict(nticks=4, range=[0,height],),),
                             width=700,
                             margin=dict(r=20, l=10, b=10, t=10))
    for ix, xy in enumerate(np.moveaxis(trace,0,-1)):
        xy = np.moveaxis(xy,-1,0)
        if is_homogenous(trace):
            xy = extract_xys_from_state(xy)
        hist, xedges, yedges = np.histogram2d(xy[:,0], xy[:,1], bins=bins)
        xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")

        xpos = xpos.ravel()
        ypos = ypos.ravel()
        dz = hist.ravel()

        fig.add_trace(go.Mesh3d(x=xpos,y=ypos,z=dz,
                           #opacity=0.3,
                           color=ryb[ix]))
    fig.show()