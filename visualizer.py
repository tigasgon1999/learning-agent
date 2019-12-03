import numpy as np
import matplotlib.pyplot as plt


# Hide axes
# ax.xaxis.set_visible(False)
# ax.yaxis.set_visible(False)

showing = False


def showTable(q):
    global showing
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    labels = [i for i in range(len(q))]
    cols = [i for i in range(len(q[0]))]
    plt.table(cellText=q, rowLabels=labels, colLabels=cols, loc='center')
    plt.pause(0.05)
    if not showing:
        plt.show()
        showing = True


saved = False


def saveTable(q):
    global saved
    if saved:
        return
    f = open('table.csv', 'w')

    for st in q:
        for a in st:
            f.write("%f," % a)
        f.write("\n")

    f.close()
    saved = True
