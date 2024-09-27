import math


class ComputeGraphExample:
    def forward(self, w0, x0, w1, x1, w2):
        """
        f(x) = 1 / (1 + e^(-(w0 * x0 + w1 * x1 + w2)))
        """
        s0 = w0 * x0
        s1 = w1 * x1
        s2 = s0 + s1
        s3 = s2 + w2
        L = 1 / (math.pow(math.e, -s3) + 1)
        return L

    def backward(self, L, w0, x0, w1, x1, w2):
        grad_l = 1.0
        grad_s3 = (1 - L) * L * grad_l
        grad_w2 = grad_s3
        grad_s2 = grad_s3
        grad_s0 = grad_s2
        grad_s1 = grad_s2
        grad_w1 = grad_s1 * x1
        grad_x1 = grad_s1 * w1
        grad_w0 = grad_s0 * x0
        grad_x0 = grad_s0 * w0
        return grad_w0, grad_x0, grad_w1, grad_x1, grad_w2


if __name__ == '__main__':
    f = ComputeGraphExample()
    w0 = 2.0
    x0 = -1.0
    w1 = -3.0
    x1 = -2.0
    w2 = -3.0
    L = f.forward(w0, x0, w1, x1, w2)
    grads = f.backward(L, w0, x0, w1, x1, w2)
    print(grads)
