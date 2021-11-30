from matplotlib import pyplot as plt

def stacked_bar(**kwargs):
    """
    x = ['a', 'b', 'c']
    y = [[1, 2, 3], [2, 3, 4]]
    x_label = 'x'
    y_label = 'y'
    z_label = ['p', 'q']
    """
    plt.figure(figsize=(10, 10))
    for i in range(len(kwargs["y"])):
        bar_kwargs = {}
        bar_kwargs["x"] = kwargs["x"]
        bar_kwargs["height"] = kwargs["y"][i]
        bar_kwargs["width"] = 0.8
        bar_kwargs["bottom"] = 0 if i == 0 else kwargs["y"][i-1]
        if "z_label" in kwargs:
            bar_kwargs["label"] = kwargs["z_label"][i]
        plt.bar(**bar_kwargs)
    plt.xticks(kwargs["x"], rotation=90)
    plt.show()