import uproot3 as uproot
import matplotlib.pyplot as plt
plt.style.use('bmh')



def load_root(file, treename="kalman"):
    with uproot.open(file) as f:
        tree = f[treename]
        print(tree)


    return tree



def hist_plot(data, bins=50, color='skyblue', edgecolor='black', xlabel="X-axis label", ylabel="Y-axis label", title="Histogram of some_variable"):
    '''
    :param data:
    :param bins:
    :param color:
    :param edgecolor:
    :param xlabel:
    :param ylabel:
    :param title:
    :return:
    '''
    plt.hist(data, bins=bins, color=color, edgecolor=edgecolor)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    h = load_root("test2.root")
    print(h.array("ori_path"))
