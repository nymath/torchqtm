# Navigation

=== "Black Scholes"
    $$
    \begin{cases}
        rf = \frac{\partial f}{\partial t} + r x \frac{\partial f}{\partial x} + \frac{1}{2} \sigma^2 x^2 \frac{\partial^2 f}{\partial x^2}\\
        f(T,x) = (x-K)^{+}
    \end{cases}
    $$

=== "Riesz  Representation Theorem"
    > Suppose $\varphi$ is a **bounded** linear functional on a Hilbert space V. Then there exists
    > a unique $h \in V$ such that
    > $$ \varphi(f) = \langle f, h \rangle$$
    > for all $f \in V$. Furthermore, $\Vert \varphi \Vert = \Vert h \Vert$.