import matplotlib.pyplot as plt


def train_acc(result, title):
    dot, general, concat = result
    plt.figure(figsize=(5, 5), dpi=300)
    plt.plot(dot, color='orange', label='Dot')
    plt.plot(dot, "o", color='orange')

    plt.plot(general, color='blue', label='General')
    plt.plot(general, "o", color='blue')

    plt.plot(concat, color='red', label='Concat')
    plt.plot(concat, "o", color='red')

    plt.xlabel('Train Epoch')
    plt.ylabel('Accuracy')
    plt.xlim(-0.5, 5.5)
    plt.ylim(0, 1)
    plt.title(title)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    LSTM_Dot_Acc = [0.6094435876141892, 0.7760855380234903, 0.889948392454621, 0.8676147822992051, 0.8897556056471705, 0.9035769367659272]
    LSTM_General_Acc = [0.6264384861786689, 0.8343071538735318, 0.8918614307747064, 0.888895479890853, 0.9152479534938901, 0.88895479890853]
    LSTM_Concat_Acc = [0.5935312611223158, 0.7026189346304426, 0.8359087673508127, 0.8223988610748606, 0.8189138687863329, 0.8719154110807925]
    GRU_Dot_Acc = [0.6765482263613715, 0.9474285205836991, 0.9722980187448096, 0.9269782892395302, 0.9504834499940681, 0.9274083521176889]
    GRU_General_Acc = [0.6397556056471705, 0.8655089571716692, 0.8901263495076521, 0.872938664135722, 0.9114070470993, 0.9031468738877684]
    GRU_Concat_Acc = []
    train_acc((LSTM_Dot_Acc, LSTM_General_Acc, LSTM_Concat_Acc), 'LSTM')

    plt.figure(figsize=(5, 5), dpi=300)
    plt.plot(GRU_Dot_Acc, color='orange', label='Dot')
    plt.plot(GRU_Dot_Acc, "o", color='orange')

    plt.plot(GRU_General_Acc, color='blue', label='General')
    plt.plot(GRU_General_Acc, "o", color='blue')

    plt.xlabel('Train Epoch')
    plt.ylabel('Accuracy')
    plt.xlim(-0.5, 5.5)
    plt.ylim(0, 1)
    plt.title('GRU')
    plt.legend()
    plt.show()
