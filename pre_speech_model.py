

def generate_data_line(xs, ys, length=2048, window=16):
    x_out = []
    y_out = []
    for i in range(0, length * window, window):
        current_x = np.asarray( xs[i: i+TIME_STEPS].reshape((-FEATURES, FEATURES))).astype('float32')
        current_y = np.asarray([ys[i + TIME_STEPS]]).astype('float32')
        x_out.append(current_x)
        y_out.append(current_y)

    return np.asarray(x_out), np.asarray(y_out).astype('float32')