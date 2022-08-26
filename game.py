import sys
import tkinter
from tkinter import Tk, Frame, Label, Button, StringVar
from tkinter.font import Font, families
from tkinter.messagebox import showinfo
from time import sleep
from threading import Thread

sys.path.append('/storage/emulated/0/Python Projects/Neural Network/')
from matrix import *
from neuralnet import *

family: str
l = [
    [0.769885, 0.452589, 0.449434, 0.529262, 0.993933, 0.686735, 0.69886, 0.133325, 0.931365],
    [0.750605, 0.377973, 0.491366, 0.256832, 0.196735, 0.980147, 0.203523, 0.357459, 0.223365],
    [0.0317631, 0.464644, 0.834493, 0.584378, 0.2426, 0.946797, 0.947822, 0.75162, 0.550059],
    [0.457008, 0.529922, 0.93092, 0.332533, 0.299807, 0.383509, 0.781967, 0.829068, 0.377442],
    [0.468702, 0.527929, 0.510767, 0.400067, 0.278534, 0.88874, 0.891433, 0.535365, 0.0854747],
    [0.87158, 0.738889, 0.442934, 0.0949454, 0.770652, 0.907578, 0.929439, 0.355029, 0.150178],
    [0.876236, 0.302851, 0.901798, 0.426295, 0.759859, 0.43172, 0.357215, 0.0923923, 0.731527],
    [0.740724, 0.874359, 0.560595, 0.118166, 0.343061, 0.088524, 0.628933, 0.743128, 0.367058],
    [0.517673, 0.634561, 0.902423, 0.603148, 0.506142, 0.641312, 0.0460813, 0.601087, 0.411963],
],
[
    [0.114045, 0.526277, 0.890397, 0.538304, 0.930134, 0.207677, 0.3491, 0.907929, 0.264182],
    [0.159269, 0.288785, 0.787955, 0.726534, 0.891826, 0.100247, 0.261008, 0.948546, 0.070584],
    [0.832222, 0.760616, 0.135485, 0.368296, 0.123906, 0.466848, 0.43409, 0.881501, 0.156606],
    [0.559442, 0.702802, 0.164539, 0.0892564, 0.816847, 0.690816, 0.979654, 0.355151, 0.62095],
    [0.187331, 0.704252, 0.528879, 0.451513, 0.863521, 0.817664, 0.239468, 0.590055, 0.70949],
    [0.339715, 0.851063, 0.658036, 0.410299, 0.683285, 0.418653, 0.545784, 0.051581, 0.542558],
    [0.0126317, 0.485671, 0.424059, 0.169237, 0.0451129, 0.126861, 0.333776, 0.134369, 0.943709],
    [0.0245919, 0.114023, 0.29886, 0.645542, 0.301354, 0.00311138, 0.174421, 0.752867, 0.866632],
    [0.992085, 0.992336, 0.456687, 0.701575, 0.332051, 0.30775, 0.359611, 0.74235, 0.991036],
],
[
    [0.871742, 0.803779, 0.707769, 0.81545, 0.828371, 0.821792, 0.11431, 0.473913, 0.123146],
    [0.117422, 0.648334, 0.876014, 0.984054, 0.640419, 0.868349, 0.440741, 0.341994, 0.2004],
    [0.748492, 0.701604, 0.94275, 0.739527, 0.479867, 0.230885, 0.782144, 0.800689, 0.53165],
    [0.310431, 0.545569, 0.00165374, 0.883831, 0.417311, 0.805433, 0.5916, 0.232761, 0.633804],
    [0.413393, 0.347072, 0.107717, 0.536539, 0.464493, 0.756052, 0.412553, 0.448547, 0.396471],
    [0.280902, 0.889288, 0.738464, 0.481302, 0.63778, 0.440068, 0.424052, 0.377308, 0.919936],
    [0.654937, 0.159452, 0.720625, 0.186587, 0.469883, 0.266194, 0.188241, 0.353714, 0.683505],
    [0.993674, 0.945314, 0.916266, 0.627478, 0.358707, 0.263338, 0.735195, 0.895246, 0.727831],
    [0.491247, 0.307799, 0.176378, 0.887717, 0.588701, 0.0656667, 0.626181, 0.0700032, 0.703447],
],
[
    [0.973004, 0.523821, 0.0638026, 0.656509, 0.517495, 0.00911667, 0.572775, 0.144973, 0.367824],
    [0.836113, 0.880168, 0.26307, 0.563944, 0.371415, 0.570869, 0.740322, 0.259133, 0.15957],
    [0.805989, 0.885314, 0.229573, 0.509436, 0.951564, 0.723628, 0.59019, 0.937749, 0.872621],
    [0.830397, 0.644559, 0.2082, 0.540486, 0.617563, 0.732021, 0.604288, 0.274072, 0.249516],
    [0.613405, 0.846846, 0.394489, 0.981228, 0.682959, 0.274658, 0.244298, 0.246903, 0.646073],
    [0.815166, 0.987225, 0.905205, 0.974736, 0.793213, 0.79052, 0.204309, 0.302649, 0.742083],
    [0.927937, 0.892839, 0.679833, 0.800558, 0.723236, 0.324392, 0.00875828, 0.263721, 0.941956],
    [0.74078, 0.86801, 0.216027, 0.990296, 0.481414, 0.0628737, 0.384785, 0.462643, 0.745833],
    [0.659443, 0.70694, 0.992735, 0.305516, 0.522107, 0.97996, 0.210721, 0.496843, 0.773173],
]
b = [
    [0.953659],
    [0.530526],
    [0.766993],
    [0.103837],
    [0.406762],
    [0.0698441],
    [0.00563576],
    [0.833057],
    [0.829703],
],
[
    [0.494095],
    [0.231967],
    [0.706238],
    [0.375735],
    [0.099921],
    [0.817932],
    [0.159868],
    [0.529596],
    [0.90722],
],
[
    [0.687219],
    [0.23084],
    [0.464875],
    [0.73998],
    [0.635298],
    [0.340538],
    [0.16072],
    [0.0850344],
    [0.882259],
],
[
    [0.617354],
    [0.517373],
    [0.523642],
    [0.372188],
    [0.511274],
    [0.308629],
    [0.571014],
    [0.119554],
    [0.800315],
]
nn = NeuralNetwork(
    [Matrix(i) for i in l], [Matrix(i) for i in b]
)

def flash(element) -> None:
    def _flash(element):
        for _ in range(2):
            element["bg"] = 'pink'
            sleep(0.25)
            element["bg"] = '#d9d9d9'
            sleep(0.25)

    Thread(target=_flash, args=(element,)).start()

    return None


class Root:
    def __init__(self):
        self.root = Tk()
        self.root.title("Tic-tac-toe")

        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        f = Frame(
            self.root
        )
        f.grid(sticky='nesw')

        self.labels: list = [[None for _ in range(3)] for _ in range(3)]
        self.buttons: list = [[None for _ in range(3)] for _ in range(3)]

        for family in families():
            font = Font(font=(family, 20))
            if font.metrics('fixed'):
                break

        for i in range(3):
            for j in range(3):
                var = StringVar()
                var.set(' ')
                self.labels[i][j] = var
                self.buttons[i][j] = Button(
                    f,
                    textvariable=var,
                    font=Font(font=(family, 20)),
                    bd=5,
                    command=lambda x=i, y=j: self.handle_turn(x, y)
                )
                self.buttons[i][j].grid(column=i, row=j, sticky='')
                f.columnconfigure(i, weight=1)
                f.rowconfigure(j, weight=1)

        self.turn = StringVar()
        self.turn.set('x')

        Label(
            self.root,
            text=' ',
            textvariable=self.turn,
            font=Font(font=(family, 20))
        ).grid(column=0, row=1)

    import tkinter
from tkinter import Tk, Frame, Label, Button, StringVar
from tkinter.font import Font, families
from tkinter.messagebox import showinfo
from time import sleep
from threading import Thread

family: str


def flash(element) -> None:
    def _flash(element):
        for _ in range(2):
            element["bg"] = 'pink'
            sleep(0.25)
            element["bg"] = '#d9d9d9'
            sleep(0.25)

    Thread(target=_flash, args=(element,)).start()

    return None


class Root:
    def __init__(self):
        self.root = Tk()
        self.root.title("Tic-tac-toe")

        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        f = Frame(
            self.root
        )
        f.grid(sticky='nesw')

        self.labels: list = [[None for _ in range(3)] for _ in range(3)]
        self.buttons: list = [[None for _ in range(3)] for _ in range(3)]

        for family in families():
            font = Font(font=(family, 20))
            if font.metrics('fixed'):
                break

        for i in range(3):
            for j in range(3):
                var = StringVar()
                var.set(' ')
                self.labels[i][j] = var
                self.buttons[i][j] = Button(
                    f,
                    textvariable=var,
                    font=Font(font=(family, 20)),
                    bd=5,
                    command=lambda x=i, y=j: self.handle_turn(x, y)
                )
                self.buttons[i][j].grid(column=i, row=j, sticky='')
                f.columnconfigure(i, weight=1)
                f.rowconfigure(j, weight=1)

        self.turn = StringVar()
        self.turn.set('x')

        Label(
            self.root,
            text=' ',
            textvariable=self.turn,
            font=Font(font=(family, 20))
        ).grid(column=0, row=1)

    def handle_turn(self, column: int, row: int, ai: bool = False):
        if self.labels[column][row].get() != ' ':
            flash(self.buttons[column][row])
            return None

        turn = self.turn.get()
        self.labels[column][row].set(turn)

        bin_field: list = [
                [
                    r.get() == turn
                    for r in c
                ]
                for c in self.labels
        ]

        if (
            [True, True, True] in bin_field
            or [True, True, True] in [[r[i] for r in bin_field] for i in range(3)]
            or all(bin_field[i][i] for i in range(3))
            or all(bin_field[i][2-i] for i in range(3))
        ):
            showinfo('Info', f'{turn} wins!')
            self.root.destroy()

        self.turn.set('o' if self.turn.get() == 'x' else 'x')

        if not ai:
            self.handle_turn(*get_coords(self.labels), True)




def get_coords(field):
        input_ = []
        for c in field:
            for r in c:
                L = r.get()
                if L == 'x':
                    input_.append([1])
                elif L == 'o':
                    input_.append([-1])
                else:
                    input_.append([0])

        output = nn.get_output(Matrix(input_))

        max = -100000

        for i in range(3):
            for j in range(3):
                if field[i][j].get() != ' ':
                    continue
                if output[3 * i + j][0] > max:
                    max = output[3 * i + j][0]
                    col = j
                    row = i

        print(field[row][col].get())

        return row, col


r = Root()
r.handle_turn(*get_coords(r.labels), True)
r.root.mainloop()


