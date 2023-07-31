import matplotlib.pyplot as plt


def read_file(dir):
    with open(dir) as f:
        data = f.readlines()

    losses = []

    for item in data:
        loss = item.split(' ')[-1][:-2]
        if loss:
            try:
                float(loss)
            except ValueError:
                continue
            if len(loss) <= 6:
                losses.append(float(loss))
    return losses


if __name__ == '__main__':
    dot = read_file('./LSTM_dot.rtf')
    general = read_file('./LSTM_genral.rtf')
    concat = read_file('./LSTM_concat.rtf')
    plt.figure(figsize=(5, 5), dpi=300)
    plt.plot(dot, color='orange', label='Dot', linewidth=0.8)

    plt.plot(general, color='blue', label='General', linewidth=0.8)

    plt.plot(concat, color='red', label='Concat', linewidth=0.8)

    plt.xlabel('Train Steps')
    plt.ylabel('Loss')
    plt.title('LSTM Loss')
    plt.legend()
    plt.show()

    dot = read_file('./GRU_dot.rtf')
    general = read_file('./GRU_general.rtf')
    plt.figure(figsize=(5, 5), dpi=300)
    plt.plot(dot, color='orange', label='Dot', linewidth=0.8)

    plt.plot(general, color='blue', label='General', linewidth=0.8)


    plt.xlabel('Train Steps')
    plt.ylabel('Loss')
    plt.title('GRU Loss')
    plt.legend()
    plt.show()
