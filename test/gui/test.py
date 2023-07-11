import tkinter as tk


# Define function to perform operations
def calculate():
    val1 = float(entry1.get())
    val2 = float(entry2.get())
    operator = operator_var.get()

    if operator == '+':
        result = val1 + val2
    elif operator == '-':
        result = val1 - val2
    elif operator == '*':
        result = val1 * val2
    elif operator == '/':
        if val2 != 0:
            result = val1 / val2
        else:
            result = "Error! Division by zero."

    result_label.config(text="Result: " + str(result))


# Create main window
root = tk.Tk()
root.title("Simple Calculator")

# Create entry widgets for input
entry1 = tk.Entry(root)
entry2 = tk.Entry(root)
entry1.pack()
entry2.pack()

# Create and set variable for operator
operator_var = tk.StringVar()
operator_var.set("+")

# Create option menu for operator
operator_menu = tk.OptionMenu(root, operator_var, "+", "-", "*", "/")
operator_menu.pack()

# Create button to perform calculation
calc_button = tk.Button(root, text="Calculate", command=calculate)
calc_button.pack()

# Create label to display result
result_label = tk.Label(root, text="Result: ")
result_label.pack()

# Start main loop
root.mainloop()
