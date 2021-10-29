import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

th = np.arange(-100, 100, 0.1)


def demo(sty):
    mpl.style.use(sty)
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.set_title('style: {!r}'.format(sty), color='C0')

    ax.plot(th, np.cos(th), 'C1', label='C1')
    ax.plot(th, np.sin(th), 'C2', label='C2')
    ax.legend()
    plt.show()


demo('default')
demo('seaborn')