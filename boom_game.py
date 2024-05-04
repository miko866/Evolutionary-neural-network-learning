# -*- coding: utf-8 -*-


import pygame
import random
    

WIDTH, HEIGHT = 900, 500

WIN = pygame.display.set_mode((WIDTH, HEIGHT))

pygame.display.set_caption("Boom master")

pygame.font.init()

WHITE = (255, 255, 255)



FPS = 80
VELOCITY = 5


BOOM_FONT = pygame.font.SysFont("comicsans", 100)   
LEVEL_FONT = pygame.font.SysFont("comicsans", 20)   



ENEMY_IMAGE  = pygame.image.load("mine.png")
ME_IMAGE = pygame.image.load("me.png")
SEA_IMAGE = pygame.image.load("sea.png")
FLAG_IMAGE = pygame.image.load("flag.png")


ENEMY_SIZE = 50
ME_SIZE = 50

ENEMY = pygame.transform.scale(ENEMY_IMAGE, (ENEMY_SIZE, ENEMY_SIZE))
ME = pygame.transform.scale(ME_IMAGE, (ME_SIZE, ME_SIZE))
SEA = pygame.transform.scale(SEA_IMAGE, (WIDTH, HEIGHT))
FLAG = pygame.transform.scale(FLAG_IMAGE, (ME_SIZE, ME_SIZE))


# vlastni udalosti
ME_HIT = pygame.USEREVENT + 1
ME_WIN = pygame.USEREVENT + 2



# mina
class Mine:
    def __init__(self):

        # random x direction
        if random.random() > 0.5:
            self.dirx = 1
        else: 
            self.dirx = -1
            
        # random y direction    
        if random.random() > 0.5:
            self.diry = 1
        else: 
            self.diry = -1

        x = random.randint(200, WIDTH - ENEMY_SIZE) 
        y = random.randint(200, HEIGHT - ENEMY_SIZE) 
        self.rect = pygame.Rect(x, y, ENEMY_SIZE, ENEMY_SIZE)
        
        self.velocity = random.randint(1, 5)
        
        
        
        

def draw_window(me, mines, level):
    WIN.blit(SEA, (0, 0))   
    
    t = LEVEL_FONT.render("level " + str(level), 1, WHITE)   
    WIN.blit(t, (10  , HEIGHT - 30))

    WIN.blit(FLAG, (WIDTH - ME_SIZE, HEIGHT - ME_SIZE - 10))    
         
    for mine in mines:
        WIN.blit(ENEMY, (mine.rect.x, mine.rect.y))
        
    WIN.blit(ME, (me.x, me.y))
        
    pygame.display.update()
    


# TODO rewrite it and improve it 
def handle_me_movement(keys_pressed, me):
    if keys_pressed[pygame.K_LEFT] and me.x - VELOCITY > 0:
        me.x -= VELOCITY
    if keys_pressed[pygame.K_RIGHT] and me.x + me.width + VELOCITY < WIDTH:
        me.x += VELOCITY
    if keys_pressed[pygame.K_UP] and me.y - VELOCITY > 0:
        me.y -= VELOCITY
    if keys_pressed[pygame.K_DOWN] and me.y + me.height + VELOCITY < HEIGHT:
        me.y += VELOCITY


    # TODO rewrite it and improve it 
def handle_mine_movement(mine):
    if mine.dirx == -1 and mine.rect.x - mine.velocity < 0:
        mine.dirx = 1
       
    if mine.dirx == 1  and mine.rect.x + mine.rect.width + mine.velocity > WIDTH:
        mine.dirx = -1

    if mine.diry == -1 and mine.rect.y - mine.velocity < 0:
        mine.diry = 1
    
    if mine.diry == 1  and mine.rect.y + mine.rect.height + mine.velocity > HEIGHT:
        mine.diry = -1
         
    mine.rect.x += mine.dirx * mine.velocity
    mine.rect.y += mine.diry * mine.velocity
    

def set_mines(num):
    l = []
    for i in range(num):
        m = Mine()
        l.append(m)
        
    return l





# TODO rewrite it and improve it 
def me_collision(me, mines):
    
    for mine in mines:
        if me.colliderect(mine.rect):
            #pygame.event.post(pygame.event.Event(ME_HIT))
            return True
    return False
            
            
     
        
        
def draw_text(text):
    t = BOOM_FONT.render(text, 1, WHITE)   
    WIN.blit(t, (WIDTH // 2  , HEIGHT // 2))     
    
    pygame.display.update()
    pygame.time.delay(1000)
    

def me_win(me):
    if me.x > WIDTH - ME_SIZE - 15 and me.y > HEIGHT - ME_SIZE -15:
        return True
    
    return False



def main():
    
    
    mines = []

    me = pygame.Rect(10, 10, ME_SIZE, ME_SIZE)
    
    clock = pygame.time.Clock()
    
    run = True
    newround = True
    level = 0
    
    while run:  
        
        clock.tick(FPS)
        
        if newround:
            newround = False
            level += 1
            me.x = 10
            me.y = 10
            mines = set_mines(level) 
            
            """# hard countdown
            draw_window(me, mines)
            draw_text("3")
            draw_window(me, mines)
            draw_text("2")
            draw_window(me, mines)
            draw_text("1")"""
    
                
        keys_pressed = pygame.key.get_pressed()

        handle_me_movement(keys_pressed, me)
        
        
        for mine in mines:
            handle_mine_movement(mine)
        
        if me_collision(me, mines):
            newround = True
            level = 0
            #draw_text("Boom !!!")"""

            
        if me_win(me): 
            newround = True
            
                
        draw_window(me, mines, level)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
    
    pygame.quit()    


if __name__ == "__main__":
    main()