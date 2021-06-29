# -*- coding: utf-8 -*-
"""
Created on Sun May 30 10:43:24 2021

@author: stefi
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from selenium import webdriver
import time
import colorama
from colorama import Fore, Style
from timeit import default_timer as timer
from sklearn.svm import SVC
      

def scacchiera(pick):
    import pygame

    import time
    
    import sys
    
    import pandas as pd
    import random 
    
    board = [['  ' for i in range(8)] for i in range(8)]
    
    ## Creates a chess piece class that shows what team a piece is on, what type of piece it is and whether or not it can be killed by another selected piece.
    class Piece:
        def __init__(self, team, type, image, killable=False):
            self.team = team
            self.type = type
            self.killable = killable
            self.image = image
    
    
    ## Creates instances of chess pieces, so far we got: pawn, king, rook and bishop
    ## The first parameter defines what team its on and the second, what type of piece it is
    
    bk = Piece('b', 'k', 'b_king.png')
    wk = Piece('w', 'k', 'w_king.png')
    wr = Piece('w', 'r', 'w_rook.png')
    
    # chess=pd.read_csv("chess.csv")
    # n=random.randint(0, len(chess.target))
    # pick=chess.iloc[n]
    
    wk_pos=(pick["wk file"]-1,9-pick["wk rank"]-1)
    wr_pos=(pick["wr file"]-1,9-pick["wr rank"]-1)
    bk_pos=(pick["bk file"]-1,9-pick["bk rank"]-1)
    
   
   
    
    
    starting_order = {(0, 0): None, (1, 0): None,
                      (2, 0): None, (3, 0): None,
                      (4, 0): None, (5, 0): None,
                      (6, 0): None, (7, 0): None,
                      (0, 1): None, (1, 1): None,
                      (2, 1): None, (3, 1): None,
                      (4, 1): None, (5, 1): None,
                      (6, 1): None, (7, 1): None,
    
                      (0, 2): None, (1, 2): None, (2, 2): None, (3, 2): None,
                      (4, 2): None, (5, 2): None, (6, 2): None, (7, 2): None,
                      (0, 3): None, (1, 3): None, (2, 3): None, (3, 3): None,
                      (4, 3): None, (5, 3): None, (6, 3): None, (7, 3): None,
                      (0, 4): None, (1, 4): None, (2, 4): None, (3, 4): None,
                      (4, 4): None, (5, 4): None, (6, 4): None, (7, 4): None,
                      (0, 5): None, (1, 5): None, (2, 5): None, (3, 5): None,
                      (4, 5): None, (5, 5): None, (6, 5): None, (7, 5): None,
    
                      (0, 6): None, (1, 6): None,
                      (2, 6): None, (3, 6): None,
                      (4, 6): None, (5, 6): None,
                      (6, 6): None, (7, 6): None,
                      (0, 7): None, (1, 7): None,
                      (2, 7): None, (3, 7): None,
                      (4, 7): None, (5, 7): None,
                      (6, 7): None, (7, 7): None,}
    
    starting_order[bk_pos]=pygame.image.load(bk.image)
    starting_order[wk_pos]=pygame.image.load(wk.image)
    starting_order[wr_pos]=pygame.image.load(wr.image)
    
    
    def create_board(board):
        board[wr_pos[1]][wr_pos[0]]=Piece('w', 'r', 'w_rook.png')
        board[bk_pos[1]][bk_pos[0]]=Piece('b', 'k', 'b_king.png')
        board[wk_pos[1]][wk_pos[0]]=Piece('w', 'k', 'w_king.png')
    
    
    
    ## returns the input if the input is within the boundaries of the board
    def on_board(position):
        if position[0] > -1 and position[1] > -1 and position[0] < 8 and position[1] < 8:
            return True
    
    
#    ## returns a string that places the rows and columns of the board in a readable manner
#    def convert_to_readable(board):
#        output = ''
#    
#        for i in board:
#            for j in i:
#                try:
#                    output += j.team + j.type + ', '
#                except:
#                    output += j + ', '
#            output += '\n'
#        return output
#    
    
#    ## resets "x's" and killable pieces
#    def deselect():
#        for row in range(len(board)):
#            for column in range(len(board[0])):
#                if board[row][column] == 'x ':
#                    board[row][column] = '  '
#                else:
#                    try:
#                        board[row][column].killable = False
#                    except:
#                        pass
#        return convert_to_readable(board)
    
    
#    ## Takes in board as argument then returns 2d array containing positions of valid moves
#    def highlight(board):
#        highlighted = []
#        for i in range(len(board)):
#            for j in range(len(board[0])):
#                if board[i][j] == 'x ':
#                    highlighted.append((i, j))
#                else:
#                    try:
#                        if board[i][j].killable:
#                            highlighted.append((i, j))
#                    except:
#                        pass
#        return highlighted
#    
#    def check_team(moves, index):
#        row, col = index
#        if moves%2 == 0:
#            if board[row][col].team == 'w':
#                return True
#        else:
#            if board[row][col].team == 'b':
#                return True
#    
#    ## This takes in a piece object and its index then runs then checks where that piece can move using separately defined functions for each type of piece.
#    def select_moves(piece, index, moves):
#        if check_team(moves, index):
#    
#            if piece.type == 'k':
#                return highlight(king_moves(index))
#    
#            if piece.type == 'r':
#                return highlight(rook_moves(index))
    
    
    
    ## Basically, check black and white pawns separately and checks the square above them. If its free that space gets an "x" and if it is occupied by a piece of the opposite team then that piece becomes killable.
    # def pawn_moves_b(index):
    #     if index[0] == 1:
    #         if board[index[0] + 2][index[1]] == '  ' and board[index[0] + 1][index[1]] == '  ':
    #             board[index[0] + 2][index[1]] = 'x '
    #     bottom3 = [[index[0] + 1, index[1] + i] for i in range(-1, 2)]
    
    #     for positions in bottom3:
    #         if on_board(positions):
    #             if bottom3.index(positions) % 2 == 0:
    #                 try:
    #                     if board[positions[0]][positions[1]].team != 'b':
    #                         board[positions[0]][positions[1]].killable = True
    #                 except:
    #                     pass
    #             else:
    #                 if board[positions[0]][positions[1]] == '  ':
    #                     board[positions[0]][positions[1]] = 'x '
    #     return board
    
    # def pawn_moves_w(index):
    #     if index[0] == 6:
    #         if board[index[0] - 2][index[1]] == '  ' and board[index[0] - 1][index[1]] == '  ':
    #             board[index[0] - 2][index[1]] = 'x '
    #     top3 = [[index[0] - 1, index[1] + i] for i in range(-1, 2)]
    
    #     for positions in top3:
    #         if on_board(positions):
    #             if top3.index(positions) % 2 == 0:
    #                 try:
    #                     if board[positions[0]][positions[1]].team != 'w':
    #                         board[positions[0]][positions[1]].killable = True
    #                 except:
    #                     pass
    #             else:
    #                 if board[positions[0]][positions[1]] == '  ':
    #                     board[positions[0]][positions[1]] = 'x '
    #     return board
    
    
#    ## This just checks a 3x3 tile surrounding the king. Empty spots get an "x" and pieces of the opposite team become killable.
#    def king_moves(index):
#        for y in range(3):
#            for x in range(3):
#                if on_board((index[0] - 1 + y, index[1] - 1 + x)):
#                    if board[index[0] - 1 + y][index[1] - 1 + x] == '  ':
#                        board[index[0] - 1 + y][index[1] - 1 + x] = 'x '
#                    else:
#                        if board[index[0] - 1 + y][index[1] - 1 + x].team != board[index[0]][index[1]].team:
#                            board[index[0] - 1 + y][index[1] - 1 + x].killable = True
#        return board
#    
#    
#    ## This creates 4 lists for up, down, left and right and checks all those spaces for pieces of the opposite team. The list comprehension is pretty long so if you don't get it just msg me.
#    def rook_moves(index):
#        cross = [[[index[0] + i, index[1]] for i in range(1, 8 - index[0])],
#                 [[index[0] - i, index[1]] for i in range(1, index[0] + 1)],
#                 [[index[0], index[1] + i] for i in range(1, 8 - index[1])],
#                 [[index[0], index[1] - i] for i in range(1, index[1] + 1)]]
#    
#        for direction in cross:
#            for positions in direction:
#                if on_board(positions):
#                    if board[positions[0]][positions[1]] == '  ':
#                        board[positions[0]][positions[1]] = 'x '
#                    else:
#                        if board[positions[0]][positions[1]].team != board[index[0]][index[1]].team:
#                            board[positions[0]][positions[1]].killable = True
#                        break
#        return board
    
    
    # ## Same as the rook but this time it creates 4 lists for the diagonal directions and so the list comprehension is a little bit trickier.
    # def bishop_moves(index):
    #     diagonals = [[[index[0] + i, index[1] + i] for i in range(1, 8)],
    #                  [[index[0] + i, index[1] - i] for i in range(1, 8)],
    #                  [[index[0] - i, index[1] + i] for i in range(1, 8)],
    #                  [[index[0] - i, index[1] - i] for i in range(1, 8)]]
    
    #     for direction in diagonals:
    #         for positions in direction:
    #             if on_board(positions):
    #                 if board[positions[0]][positions[1]] == '  ':
    #                     board[positions[0]][positions[1]] = 'x '
    #                 else:
    #                     if board[positions[0]][positions[1]].team != board[index[0]][index[1]].team:
    #                         board[positions[0]][positions[1]].killable = True
    #                     break
    #     return board
    
    
    # ## applies the rook moves to the board then the bishop moves because a queen is basically a rook and bishop in the same position.
    # def queen_moves(index):
    #     board = rook_moves(index)
    #     board = bishop_moves(index)
    #     return board
    
    
    # ## Checks a 5x5 grid around the piece and uses pythagoras to see if if a move is valid. Valid moves will be a distance of sqrt(5) from centre
    # def knight_moves(index):
    #     for i in range(-2, 3):
    #         for j in range(-2, 3):
    #             if i ** 2 + j ** 2 == 5:
    #                 if on_board((index[0] + i, index[1] + j)):
    #                     if board[index[0] + i][index[1] + j] == '  ':
    #                         board[index[0] + i][index[1] + j] = 'x '
    #                     else:
    #                         if board[index[0] + i][index[1] + j].team != board[index[0]][index[1]].team:
    #                             board[index[0] + i][index[1] + j].killable = True
    #     return board
    
    
    WIDTH = 800
    
    WIN = pygame.display.set_mode((WIDTH, WIDTH))
    
    """ This is creating the window that we are playing on, it takes a tuple argument which is the dimensions of the window so in this case 800 x 800px
    """
    
    pygame.display.set_caption("Chess")
    WHITE = (255, 255, 255)
    GREY = (128, 128, 128)
    YELLOW = (204, 204, 0)
    BLUE = (50, 255, 255)
    BLACK = (0, 0, 0)
    
    
    class Node:
        def __init__(self, row, col, width):
            self.row = row
            self.col = col
            self.x = int(row * width)
            self.y = int(col * width)
            self.colour = WHITE
            self.occupied = None
    
        def draw(self, WIN):
            pygame.draw.rect(WIN, self.colour, (self.x, self.y, WIDTH / 8, WIDTH / 8))
    
        def setup(self, WIN):
            if starting_order[(self.row, self.col)]:
                if starting_order[(self.row, self.col)] == None:
                    pass
                else:
                    WIN.blit(starting_order[(self.row, self.col)], (self.x, self.y))
    
            """
            For now it is drawing a rectangle but eventually we are going to need it
            to use blit to draw the chess pieces instead
            """
    
    
    def make_grid(rows, width):
        grid = []
        gap = WIDTH // rows
        print(gap)
        for i in range(rows):
            grid.append([])
            for j in range(rows):
                node = Node(j, i, gap)
                grid[i].append(node)
                if (i+j)%2 ==1:
                    grid[i][j].colour = GREY
        return grid
    """
    This is creating the nodes thats are on the board(so the chess tiles)
    I've put them into a 2d array which is identical to the dimesions of the chessboard
    """
    
    
    def draw_grid(win, rows, width):
        gap = width // 8
        for i in range(rows):
            pygame.draw.line(win, BLACK, (0, i * gap), (width, i * gap))
            for j in range(rows):
                pygame.draw.line(win, BLACK, (j * gap, 0), (j * gap, width))
    
        """
        The nodes are all white so this we need to draw the grey lines that separate all the chess tiles
        from each other and that is what this function does"""
    
    
    def update_display(win, grid, rows, width):
        for row in grid:
            for spot in row:
                spot.draw(win)
                spot.setup(win)
        draw_grid(win, rows, width)
        pygame.display.update()
    
    
    def Find_Node(pos, WIDTH):
        interval = WIDTH / 8
        y, x = pos
        rows = y // interval
        columns = x // interval
        return int(rows), int(columns)
    
    
#    def display_potential_moves(positions, grid):
#        for i in positions:
#            x, y = i
#            grid[x][y].colour = BLUE
#            """
#            Displays all the potential moves
#            """
    
    
#    def Do_Move(OriginalPos, FinalPosition, WIN):
#        starting_order[FinalPosition] = starting_order[OriginalPos]
#        starting_order[OriginalPos] = None
#    
#    
#    def remove_highlight(grid):
#        for i in range(len(grid)):
#            for j in range(len(grid[0])):
#                if (i+j)%2 == 0:
#                    grid[i][j].colour = WHITE
#                else:
#                    grid[i][j].colour = GREY
#        return grid
    """this takes in 2 co-ordinate parameters which you can get as the position of the piece and then the position of the node it is moving to
    you can get those co-ordinates using my old function for swap"""
    
    create_board(board)
    
    
    def main(WIN, WIDTH):
        moves = 0
        selected = False
        piece_to_move=[]
        grid = make_grid(8, WIDTH)
        
        while True:
            pygame.time.delay(50) ##stops cpu dying
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    
                    return
    
                """This quits the program if the player closes the window"""
    
#                if event.type == pygame.MOUSEBUTTONDOWN:
#                    pos = pygame.mouse.get_pos()
#                    y, x = Find_Node(pos, WIDTH)
#                    if selected == False:
#                        try:
#                            possible = select_moves((board[x][y]), (x,y), moves)
#                            for positions in possible:
#                                row, col = positions
#                                grid[row][col].colour = BLUE
#                            piece_to_move = x,y
#                            selected = True
#                        except:
#                            piece_to_move = []
#                            print('Can\'t select')
#                        #print(piece_to_move)
#    
#                    else:
#                        try:
#                            if board[x][y].killable == True:
#                                row, col = piece_to_move ## coords of original piece
#                                board[x][y] = board[row][col]
#                                board[row][col] = '  '
#                                deselect()
#                                remove_highlight(grid)
#                                Do_Move((col, row), (y, x), WIN)
#                                moves += 1
#                                print(convert_to_readable(board))
#                            else:
#                                deselect()
#                                remove_highlight(grid)
#                                selected = False
#                                print("Deselected")
#                        except:
#                            if board[x][y] == 'x ':
#                                row, col = piece_to_move
#                                board[x][y] = board[row][col]
#                                board[row][col] = '  '
#                                deselect()
#                                remove_highlight(grid)
#                                Do_Move((col, row), (y, x), WIN)
#                                moves += 1
#                                print(convert_to_readable(board))
#                            else:
#                                deselect()
#                                remove_highlight(grid)
#                                selected = False
#                                print("Invalid move")
#                        selected = False
    
                update_display(WIN, grid, 8, WIDTH)
    
    
    main(WIN, WIDTH)

    
def posizioni():
    wk_file = 15
    wk_rank = 15
    wr_file = 15
    wr_rank = 15
    bk_file = 15
    bk_rank = 15
    print("Dammi file e rank di re bianco, torre bianca e re nero: ")
    while(wk_file > 8 or wk_file < 0):
        wk_file = int(input("Re bianco file: "))
    while(wk_rank > 8 or wk_file < 0):
        wk_rank = int(input("Re bianco rank: "))
    while(wr_file > 8 or wk_file < 0):    
        wr_file = int(input("Torre bianca file: "))
    while(wr_rank > 8 or wk_file < 0):    
        wr_rank = int(input("Torre bianca rank: "))
    while(bk_file > 8 or wk_file < 0):    
        bk_file = int(input("Re nero file: "))
    while(bk_rank > 8 or wk_file < 0):    
        bk_rank = int(input("Re nero rank: "))
    
    #wk_file=int(input("Re bianco file:"))
    #wk_rank=int(input("Re bianco rank:"))
    #wr_file=int(input("Torre bianca file:"))
    #wr_rank=int(input("Torre bianca rank:"))
    #bk_file=int(input("Re nero file:"))
    #bk_rank=int(input("Re nero rank:"))
    
    while ((wk_file == wr_file and wk_rank == wr_rank) or
           (wk_file == bk_file and wk_rank == bk_rank) or
           (wr_file == bk_file and wr_rank == bk_rank)):
        print("Posizione non valida, re bianco e re nero in posizioni adiacenti. Reiniserire le posizioni:")
        while(wk_file > 8 or wk_file < 0):
            wk_file = int(input("Re bianco file: "))
        while(wk_rank > 8 or wk_file < 0):
            wk_rank = int(input("Re bianco rank: "))
        while(wr_file > 8 or wk_file < 0):    
            wr_file = int(input("Torre bianca file: "))
        while(wr_rank > 8 or wk_file < 0):    
            wr_rank = int(input("Torre bianca rank: "))
        while(bk_file > 8 or wk_file < 0):    
            bk_file = int(input("Re nero file: "))
        while(bk_rank > 8 or wk_file < 0):    
            bk_rank = int(input("Re nero rank: "))
    
    while abs(wk_file-bk_file)<=1 and abs(wk_rank-bk_rank)<=1:
        print("Posizione non valida, re bianco e re nero in posizioni adiacenti. Reiniserire le posizioni:")
        wk_file=int(input("Re bianco file:"))
        wk_rank=int(input("Re bianco rank:"))
        bk_file=int(input("Re nero file:"))
        bk_rank=int(input("Re nero rank:"))
        
    
    if wk_file>4 and wk_rank<=4:
        wk_file_s=9-wk_file
        wr_file_s=9-wr_file
        bk_file_s=9-bk_file
        wk_rank_s=wk_rank
        wr_rank_s=wr_rank
        bk_rank_s=bk_rank
    elif wk_file>4 and wk_rank>4:
        wk_file_s=9-wk_file
        wr_file_s=9-wr_file
        bk_file_s=9-bk_file
        wk_rank_s=9-wk_rank
        wr_rank_s=9-wr_rank
        bk_rank_s=9-bk_rank
    elif wk_file<=4 and wk_rank>4:
        wk_rank_s=9-wk_rank
        wr_rank_s=9-wr_rank
        bk_rank_s=9-bk_rank
        wk_file_s=wk_file
        wr_file_s=wr_file
        bk_file_s=bk_file
    else:
        wk_file_s = wk_file
        wk_rank_s = wk_rank
        wr_file_s = wr_file
        wr_rank_s = wr_rank
        bk_file_s = bk_file
        bk_rank_s = bk_rank
        
        
    if wk_file_s<wk_rank_s:
        t=wk_rank_s
        wk_rank_s=wk_file_s
        wk_files_s=t
        t=wr_rank_s
        wr_rank_s=wr_file_s
        wr_file_s=t
        t=bk_rank_s
        bk_rank_s=bk_file_s
        bk_file_s=t

    return wk_file, wk_rank, wr_file, wr_rank, bk_file, bk_rank, wk_file_s, wk_rank_s, wr_file_s, wr_rank_s, bk_file_s, bk_rank_s


def Pca(X):
    p = False
    p = input("Vuoi vedere il grafico PCA 2D? Y/N: ")
    if  p=="Y" or p=="y" or p=="yes" or p=="Yes":
        scaler = StandardScaler()
        X_scaler = scaler.fit_transform(X)
        pca = PCA()
        pca.fit(X_scaler)
        X_pca = pca.transform(X_scaler)
        plt.figure()
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Numero di componenti')
        plt.ylabel('Varianza cumulativa')
        plt.show()
        plt.figure()
        plt.figure(figsize=(10,8))
        plt.scatter(X_pca[y == -1][:,0],X_pca[y == -1][:,1], label=1)
        plt.scatter(X_pca[y == 0][:,0],X_pca[y == 0][:,1], label=2)
        plt.scatter(X_pca[y == 1][:,0],X_pca[y == 1][:,1], label=3)
        plt.scatter(X_pca[y == 2][:,0],X_pca[y == 2][:,1], label=4)
        plt.scatter(X_pca[y == 3][:,0],X_pca[y == 3][:,1], label=5)
        plt.scatter(X_pca[y == 4][:,0],X_pca[y == 4][:,1], label=6)
        plt.scatter(X_pca[y == 5][:,0],X_pca[y == 5][:,1], label=7)
        plt.scatter(X_pca[y == 6][:,0],X_pca[y == 6][:,1], label=8)
        plt.scatter(X_pca[y == 7][:,0],X_pca[y == 7][:,1], label=9)
        plt.scatter(X_pca[y == 8][:,0],X_pca[y == 8][:,1], label=10)
        plt.scatter(X_pca[y == 9][:,0],X_pca[y == 9][:,1], label=11)
        plt.scatter(X_pca[y == 10][:,0],X_pca[y == 10][:,1], label=12)
        plt.scatter(X_pca[y == 11][:,0],X_pca[y == 11][:,1], label=13)
        plt.scatter(X_pca[y == 12][:,0],X_pca[y == 12][:,1], label=14)
        plt.scatter(X_pca[y == 13][:,0],X_pca[y == 13][:,1], label=15)
        plt.scatter(X_pca[y == 14][:,0],X_pca[y == 14][:,1], label=16)
        plt.scatter(X_pca[y == 15][:,0],X_pca[y == 15][:,1], label=17)
        plt.scatter(X_pca[y == 16][:,0],X_pca[y == 16][:,1], label=18)
        plt.legend(("-1","0","1","2","3","4","5","6","7",
                    "8","9","10","11","12","13","14","15","16"),
                   bbox_to_anchor=(0.84, 0.65), ncol=2)
        plt.title("Grafico PCA dati")
        plt.pause(0.01)
        
    del p
        
        
    p = False    
    p = input("Vuoi vedere il grafico PCA 3D? Y/N: ")
    if  p=="Y" or p=="y" or p=="yes" or p=="Yes":
        scaler = StandardScaler()
        X_scaler = scaler.fit_transform(X)
        pca = PCA()
        pca.fit(X_scaler)
        X_pca = pca.transform(X_scaler)
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Numero di componenti')
        plt.ylabel('Varianza cumulativa')
        plt.show()
        fig = plt.figure()
        ax = Axes3D(fig)
        plt.figure(figsize=(10,8))
        ax.scatter(X_pca[y == -1][:,0],X_pca[y == -1][:,1],X_pca[y == -1][:,2], label=1)
        ax.scatter(X_pca[y == 0][:,0],X_pca[y == 0][:,1],X_pca[y == -0][:,2], label=2)
        ax.scatter(X_pca[y == 1][:,0],X_pca[y == 1][:,1],X_pca[y == 1][:,2], label=3)
        ax.scatter(X_pca[y == 2][:,0],X_pca[y == 2][:,1],X_pca[y == 2][:,2], label=4)
        ax.scatter(X_pca[y == 3][:,0],X_pca[y == 3][:,1],X_pca[y == 3][:,2], label=5)
        ax.scatter(X_pca[y == 4][:,0],X_pca[y == 4][:,1],X_pca[y == 4][:,2], label=6)
        ax.scatter(X_pca[y == 5][:,0],X_pca[y == 5][:,1],X_pca[y == 5][:,2], label=7)
        ax.scatter(X_pca[y == 6][:,0],X_pca[y == 6][:,1],X_pca[y == 6][:,2], label=8)
        ax.scatter(X_pca[y == 7][:,0],X_pca[y == 7][:,1],X_pca[y == 7][:,2], label=9)
        ax.scatter(X_pca[y == 8][:,0],X_pca[y == 8][:,1],X_pca[y == 8][:,2], label=10)
        ax.scatter(X_pca[y == 9][:,0],X_pca[y == 9][:,1],X_pca[y == 9][:,2], label=11)
        ax.scatter(X_pca[y == 10][:,0],X_pca[y == 10][:,1],X_pca[y == 10][:,2], label=12)
        ax.scatter(X_pca[y == 11][:,0],X_pca[y == 11][:,1],X_pca[y == 11][:,2], label=13)
        ax.scatter(X_pca[y == 12][:,0],X_pca[y == 12][:,1],X_pca[y == 12][:,2], label=14)
        ax.scatter(X_pca[y == 13][:,0],X_pca[y == 13][:,1],X_pca[y == 13][:,2], label=15)
        ax.scatter(X_pca[y == 14][:,0],X_pca[y == 14][:,1],X_pca[y == 14][:,2], label=16)
        ax.scatter(X_pca[y == 15][:,0],X_pca[y == 15][:,1],X_pca[y == 15][:,2], label=17)
        ax.scatter(X_pca[y == 16][:,0],X_pca[y == 16][:,1],X_pca[y == 16][:,2], label=18)
        ax.legend(("-1","0","1","2","3","4","5","6","7",
                    "8","9","10","11","12","13","14","15","16"),
                   bbox_to_anchor=(1.04, 0.85), ncol=2)
        ax.set_title("Grafico PCA dati")
        plt.pause(0.01)
    del p
    
def lichess(wk_file,wk_rank,wr_file,wr_rank,bk_file,bk_rank):

    A = [[0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]]
    
    # wk_file_a = 8- wk_file
    # wk_rank_a = wk_rank - 1
    # wr_file_a = 8- wr_file
    # wr_rank_a = wr_rank - 1
    # bk_file_a = 8- bk_file
    # bk_rank_a = bk_rank - 1
    
    
    wk_file_a = wk_file -1
    wk_rank_a = 8 - wk_rank
    wr_file_a = wr_file -1
    wr_rank_a = 8 - wr_rank
    bk_file_a = bk_file -1
    bk_rank_a = 8 - bk_rank
    
    
    # A[wk_file_a][wk_rank_a] = 'wk'
    # A[wr_file_a][wr_rank_a] = 'wr'
    # A[bk_file_a][bk_rank_a] = 'bk'
    
    A[wk_rank_a][wk_file_a] = 'wk'
    A[wr_rank_a][wr_file_a] = 'wr'
    A[bk_rank_a][bk_file_a] = 'bk'
    
    print(A)
    
    totale = []
    for i in range(0,8,1):
        contatore = 0
        for j in range(0,8,1):
            if (A[i][j] == 0):
                contatore += 1
                #print("Contatore = ",contatore)
            else:
                if (contatore !=0):
                    totale.append(str(contatore))
                contatore = 0
                if (A[i][j] == 'bk'):
                    totale.append('k')
                elif (A[i][j] == 'wk'):
                    totale.append('K')
                elif (A[i][j] == 'wr'):
                    totale.append('R')
        if contatore != 0: 
            totale.append(str(contatore))
        if i!=7:
            totale.append('/')
    
    stringa = totale[0]
    for i in range(1,len(totale)):
        stringa = stringa + totale[i]
    
    if (wr_file == bk_file or wr_rank == bk_rank):
        sito_web = 'https://lichess.org/editor/' + stringa + '_b_-_-_0_1'
    else:
        sito_web = 'https://lichess.org/editor/' + stringa + '_w_-_-_0_1'
    
    driver = webdriver.Chrome("/home/flost/Documents/Università/Algoritmi e Applicazioni - Piccialli/chromedriver")
    driver.get(sito_web)
    
    if (wr_file == bk_file or wr_rank == bk_rank):
        analisi ='/analysis/' + stringa + '_b_-_-_0_1'
    else:
        analisi ='/analysis/' + stringa + '_w_-_-_0_1'
    
    class_select = driver.find_element_by_xpath("//a[@href='{}' and text()='Analysis board']".format(analisi))
    class_select.click()
    
    time.sleep(10)
    
    prof=driver.find_element_by_xpath("//a[@title='Go deeper']")
    prof.click()
    
    time.sleep(10)
    
    val = driver.find_elements_by_xpath('//*[@id="main-wrap"]/main/div[3]/div[1]/pearl')
    val=val[0].text
    # driver.find_element_by_xpath("//div[contains(@class, 'close')]")
    # driver.click()
    return val




chess=pd.read_csv(r'chess.csv')


fraction=float(input(f"{Fore.MAGENTA}Che percentuale del dataset vuoi?: {Style.RESET_ALL}"))

X = chess.sample(frac=fraction)
y=X["target"] 
X = X.drop("target", axis=1)

p=input(f"{Fore.MAGENTA}Vuoi i grafici pca? Y/N: {Style.RESET_ALL}")
if  p=="Y" or p=="y" or p=="yes" or p=="Yes":
    del p
    Pca(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=42)

p=input(f"{Fore.MAGENTA}Vuoi fittare il modello? Y/N: {Style.RESET_ALL}")
if  p=="Y" or p=="y" or p=="yes" or p=="Yes":
    del p
    xgb_tuned=xgb.XGBClassifier(max_depth=20, #40
                                objective='multi:softmax',
                                n_estimators=100, #1000
                                num_classes=18,
                                learning_rate=0.3,
                                colsamples_bytree=0.7)
    print(f"{Fore.MAGENTA}Ora inizio a fittare il modello e ci metto:{Style.RESET_ALL}")
    start = timer()
    xgb_tuned.fit(X_train,y_train , eval_set=[(X_train, y_train), (X_test, y_test)])
    print(f"{Fore.MAGENTA}Tempo di fit:{Style.RESET_ALL}", timer()-start)
    y_pred = xgb_tuned.predict(X_test)
    xgb_tuned.save_model("model_sklearn.json")
    


else:
    xgb_tuned=xgb.XGBClassifier()
    xgb_tuned.load_model("model_sklearn.json")
    y_pred = xgb_tuned.predict(X_test)
    
    

acc=accuracy_score(y_test,y_pred)
print(f"{Fore.MAGENTA}Accuracy score : %f {Style.RESET_ALL}" % (acc))

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

pred = y_pred
pred_prob = xgb_tuned.predict_proba(X_test)
fpr = {}
tpr = {}
thresh ={}
n_class=18

for i in range(n_class):    
    fpr[i], tpr[i], thresh[i] = roc_curve(y_test, pred_prob[:,i], pos_label=i)
    
    
# plotting    
plt.plot(fpr[0], tpr[0], linestyle='--',color='orange', label='Class 0 vs Rest')
plt.plot(fpr[1], tpr[1], linestyle='--',color='yellow', label='Class 1 vs Rest')
plt.plot(fpr[2], tpr[2], linestyle='--',color='blue', label='Class 2 vs Rest')
plt.plot(fpr[3], tpr[3], linestyle='--',color='indigo', label='Class 2 vs Rest')
plt.plot(fpr[4], tpr[4], linestyle='--',color='black', label='Class 2 vs Rest')
plt.plot(fpr[5], tpr[5], linestyle='--',color='lightcoral', label='Class 2 vs Rest')
plt.plot(fpr[6], tpr[6], linestyle='--',color='darkkhaki', label='Class 2 vs Rest')
plt.plot(fpr[7], tpr[7], linestyle='--',color='gold', label='Class 2 vs Rest')
plt.plot(fpr[8], tpr[8], linestyle='--',color='peru', label='Class 2 vs Rest')
plt.plot(fpr[9], tpr[9], linestyle='--',color='crimson', label='Class 2 vs Rest')
plt.plot(fpr[10], tpr[10], linestyle='--',color='lime', label='Class 2 vs Rest')
plt.plot(fpr[11], tpr[11], linestyle='--',color='olive', label='Class 2 vs Rest')
plt.plot(fpr[12], tpr[12], linestyle='--',color='red', label='Class 2 vs Rest')
plt.plot(fpr[13], tpr[13], linestyle='--',color='pink', label='Class 2 vs Rest')
plt.plot(fpr[14], tpr[14], linestyle='--',color='yellowgreen', label='Class 2 vs Rest')
plt.plot(fpr[15], tpr[15], linestyle='--',color='greenyellow', label='Class 2 vs Rest')
plt.plot(fpr[16], tpr[16], linestyle='--',color='grey', label='Class 2 vs Rest')
plt.plot(fpr[17], tpr[17], linestyle='--',color='cadetblue', label='Class 2 vs Rest')

plt.title('Multiclass ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.show()
plt.savefig('Multiclass ROC',dpi=300);





#param_grid = {'C': [0.1, 1, 10, 100], 
#              'gamma': [1, 0.1, 0.01, 0.001],
#              'kernel': ['poly']} 
#
#grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
#grid.fit(X_train, y_train)
#print("best poly params:" ,grid.best_params_)
#print("best poly estimator:",grid.best_estimator_)
#y_pred = grid.predict(X_test)
#
#
#acc=accuracy_score(y_test,y_pred)
#print(f"{Fore.MAGENTA}Accuracy score poly : %f {Style.RESET_ALL}" % (acc))






p=input(f"{Fore.MAGENTA}Vuoi provare una posizione casuale? Y/N: {Style.RESET_ALL}")    
while  p=="Y" or p=="y" or p=="yes" or p=="Yes":
    n=random.randint(0, len(y_test))
    pick=X_test.iloc[n]
    print(pick)
    if (y_pred[n]==-1):
        print(f"{Fore.MAGENTA}La posizione scelta casualmente porta al pareggio{Style.RESET_ALL}")
    else:
        print(f"{Fore.MAGENTA}Dalla posizione scelta casualmente ci vogliono %d passi per il matto{Style.RESET_ALL}" % (y_pred[n]))
    
    scacchiera(pick)

    scelta = input(f"{Fore.MAGENTA}Vuoi provare cosa ti dice lichess? Y/N:{Style.RESET_ALL}")
    if (scelta == "Y" or scelta == "y" or scelta == "yes" or scelta == "Yes"):
        del scelta
        wk_file=pick["wk file"]
        wk_rank=pick["wk rank"]
        wr_file=pick["wr file"]
        wr_rank=pick["wr rank"]
        bk_file=pick["bk file"]
        bk_rank=pick["bk rank"]
        val=lichess(wk_file,wk_rank,wr_file,wr_rank,bk_file,bk_rank)
        if(val=='0.0'):
            print(f"{Fore.MAGENTA}Secondo Lichess la posizione scelta porta al pareggio{Style.RESET_ALL}")
        else:
            val=int(val[1:])
            print(f"{Fore.MAGENTA}Secondo Lichess dalla posizione scelta casualmente ci vogliono %d passi per il matto{Style.RESET_ALL}" %(val))
            
        
        if (y_test[n] != val):
            chess.iloc[n][6]=val
        
            
            
            
        
    p=input(f"{Fore.MAGENTA}Vuoi provare un'altra posizione casuale? Y/N:{Style.RESET_ALL}")

del p
print("\n")

p=input(f"{Fore.MAGENTA}Vuoi provare una posizione a tua scelta? Y/N:{Style.RESET_ALL}")    
while  p=="Y" or p=="y" or p=="yes" or p=="Yes":
    
    wk_file, wk_rank, wr_file, wr_rank, bk_file, bk_rank, wk_file_s, wk_rank_s, wr_file_s, wr_rank_s, bk_file_s, bk_rank_s = posizioni()        

    d1 ={'wk file': [wk_file_s], 'wk rank': [wk_rank_s], 'wr file': [wr_file_s], 'wr rank': [wr_rank_s], 'bk file': [bk_file_s], 'bk rank': [bk_rank_s]}
    choice1 = pd.DataFrame(data=d1)
    position_pred=xgb_tuned.predict(choice1)
    print("Dalla posizione scelta ci vogliono %d passi per il matto" % (position_pred))
    d ={'wk file': wk_file, 'wk rank': wk_rank, 'wr file': wr_file, 'wr rank': wr_rank, 'bk file': bk_file, 'bk rank': bk_rank}
    choice = pd.Series(data=d)
    scacchiera(choice)        
    
    scelta = input(f"{Fore.MAGENTA}Vuoi provare cosa ti dice lichess? Y/N:")
    if (scelta == "Y" or scelta == "y" or scelta == "yes" or scelta == "Yes"):
        del scelta
        lots=lichess(wk_file,wk_rank,wr_file,wr_rank,bk_file,bk_rank)
        
        
    
    p=input(f"{Fore.MAGENTA}Vuoi provare un'altra posizione? Y/N:{Style.RESET_ALL}")

del p














