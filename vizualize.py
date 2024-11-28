import torch
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
from map import Map  # Importujesz swoją klasę Map

# Wczytaj model
model = torch.load('swarm_ai_model.pth')

# Stwórz mapę (np. rozmiar 200x200, 50 postaci)
map = Map(200, 200, 50, 3)  # Możesz zmienić parametry mapy
map.reset_map()
# Funkcja, która zaktualizuje widok mapy
def update_map_display():
    # Tu powinieneś zaimplementować kod do wyświetlania aktualnego stanu mapy
    # Na przykład możesz wyświetlić prostą macierz w oknie tekstowym
    display_text = ""
    for y in range(map.height):
        for x in range(map.width):
            char_id = map.grid[x, y]
            display_text += f"{char_id if char_id != -1 else '.'} "
        display_text += "\n"

    map_display.delete(1.0, tk.END)  # Usuń poprzedni tekst
    map_display.insert(tk.END, display_text)  # Wstaw nowy stan mapy

# Funkcja, która uruchamia model na danej postaci
def run_model():
    character_id = int(character_id_entry.get())  # Pobierz ID postaci z interfejsu

    # Zbierz dane wejściowe (np. okolice postaci i jej dane)
    surroundings = map.get_surroundings(character_id)
    surroundings_flat = surroundings.flatten()
    char_data = map.find_character_by_id(character_id)[3:6]  # Wybierz dane postaci
    input_data = torch.tensor(np.concatenate([surroundings_flat, char_data]), dtype=torch.float32)

    # Przewidywanie akcji przez model
    with torch.no_grad():
        output = model(input_data)  # Model zwraca wektor z wartościami dla 4 ruchów
        action = output.argmax().item()  # Wybieramy ruch o najwyższej wartości

    # Zrób ruch w zależności od przewidywanej akcji
    possible_moves = map.possible_moves(character_id)
    if possible_moves[action]:
        if action == 0:  # Góra
            map.move_character(character_id, -1, 0)
        elif action == 1:  # Dół
            map.move_character(character_id, 1, 0)
        elif action == 2:  # Lewo
            map.move_character(character_id, 0, -1)
        elif action == 3:  # Prawo
            map.move_character(character_id, 0, 1)

    # Zaktualizuj mapę po ruchu
    update_map_display()
    messagebox.showinfo("Akcja wykonana", f"Model wybrał akcję: {['Góra', 'Dół', 'Lewo', 'Prawo'][action]}")

# Tworzenie GUI
root = tk.Tk()
root.title("AI Map Simulator")

# Miejsce do wyświetlania mapy
map_display = tk.Text(root, width=60, height=20)
map_display.pack()

# Etykieta i pole tekstowe do podania ID postaci
character_id_label = tk.Label(root, text="Podaj ID postaci:")
character_id_label.pack()

character_id_entry = tk.Entry(root)
character_id_entry.pack()

# Przycisk do uruchomienia modelu i wykonania ruchu
run_button = tk.Button(root, text="Uruchom model", command=run_model)
run_button.pack()

# Inicjalizuj mapę i wyświetl początkowy stan
update_map_display()

# Uruchom GUI
root.mainloop()
