#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 14:18:57 2025

@author: michaelbridgnell
"""

import numpy as np
import ultralytics
from ultralytics import YOLO
import random
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import mss
import os

#Troop Placements

Arena_Opponent = np.zeros((32,18))
Arena_Opponent[0, 5:12] = 1
for i in range(1,15):
    Arena_Opponent[i, :] = 1

Arena_Player = np.zeros((32,18))
Arena_Player[31, 5:12] = 1
for i in range(18,31):
    Arena_Player[i, :] = 1
    
#Spell Placements

Arena_Spell = np.zeros((32,18))
for i in range(0,32):
    Arena_Spell[i,:] = 1


class ArenaEnv:
    def __init__(self, rows=32, cols=18):
        self.rows, self.cols = rows, cols
        
        # Reference your existing arrays directly
        self.player_grid = Arena_Player
        self.opponent_grid = Arena_Opponent
        self.spell_grid = Arena_Spell
        
        self.reset()

        #For example: cbot would recognize say Knight = ArenaEnv()

    def reset(self):
        # Initialize the arena state: 0 = empty, >0 = card placed
        self.grid = np.zeros((self.rows, self.cols), dtype=int)
        return self.grid

    def is_valid_action(self, action_type, row, col):
        """
        action_type: 'troop' or 'spell'
        """
        if action_type == 'troop':
            # Can place only if the player zone allows it and the cell is empty
            return self.player_grid[row, col] == 1 
        elif action_type == 'spell':
            # Can place anywhere the spell grid allows
            return self.spell_grid[row, col] == 1 
        return False

    def step(self, action_type, card_id, row, col):
        """
        Places a card if valid.
        Returns: grid, reward, done
        """
        if self.is_valid_action(action_type, row, col):
            self.grid[row, col] = card_id  # store card_id directly
            reward = 1  # placeholder
        else:
            reward = -1  # invalid placement

        done = False  # placeholder
        return self.grid, reward, done
    
# Load YOLO
model = YOLO("/Users/michaelbridgnell/Documents/runs/detect/train2/weights/best.pt")

# Init screen capture
sct = mss.mss()
monitor = {"top": 40, "left": 20, "width": 430, "height": 880}  # full screen
save_dir = r"/Users/michaelbridgnell/Documents/ClashData"
video_location = os.path.join(save_dir, "LiveClash.avi")
out = cv2.VideoWriter(video_location, cv2.VideoWriter_fourcc(*'XVID'), 20, (monitor['width'], monitor['height']))

while True:
    # Grab screenshot
   screenshot = np.array(sct.grab(monitor))
   frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
   out.write(frame)  # save frame to video
   filename = os.path.join(save_dir, "LiveClash.avi")
   if cv2.waitKey(1) & 0xFF == ord('q'):
            break
   print(f"Saved to {filename}")


    # Run YOLO
   results = model.predict(frame, verbose=False)

    # Draw detections
   for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show live window
        cv2.imshow("YOLO Detection", frame)

    # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
out.release()
cv2.destroyAllWindows()
print("Recording finished and video saved.")

    

# Neural network: input = grid, output = Q-values for actions
class DQN(nn.Module):
    def __init__(self, rows, cols, num_cards):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(rows * cols, 256),
            nn.ReLU(),
            nn.Linear(256, num_cards * rows * cols)  # each (card, row, col) action
        )

    def forward(self, x):
        return self.fc(x)

# Example usage
env = ArenaEnv()
rows, cols = env.rows, env.cols
num_cards = 5  # suppose we have 5 types of cards

model = DQN(rows, cols, num_cards)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()





pyautogui.click(x, y)