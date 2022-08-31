import os
import sys
import json
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
data = json.load(
    open('/storage/emulated/0/C++ Projects/tic-tac-toe_AI/data/17500.json')
)


#print(json.dumps(data["layers"], indent=2))

nn = NeuralNetwork(
    [Matrix(i) for i in data["layers"]], [Matrix(i) for i in data["biases"]]
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
        elif all(i.get() != ' ' for r in self.labels for i in r):
            showinfo('Info', 'Draw!')
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

        return row, col


r = Root()
r.handle_turn(*get_coords(r.labels), True)
r.root.mainloop()


