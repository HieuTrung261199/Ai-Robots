import pygame
import math
import numpy as np

WHITE_COLOR = (255, 255, 255, 255)

class Envir:
    def __init__(self, dim):
        #colors
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.red = (255, 0, 0)
        self.green = (0, 255, 0)
        self.blue = (0, 0, 255)
        #self.yellow = (255, 255, 0)
        #map dims
        self.height = dim[0]
        self.width = dim[1]
        #window settings
        pygame.display.set_caption("Robott ")
        self.map = pygame.display.set_mode((self.width, self.height))
        self.track = pygame.image.load('track.png').convert_alpha()
        self.track_copy = self.track.copy()
        self.font = pygame.font.SysFont("arial",30)
        #self.track = pygame.transform.scale(self.track, (self.width, self.height))
        self.text = self.font.render('default', True, self.black, self.white)
        self.textRect = self.text.get_rect()
        self.textRect.center = (dim[1] - 700, dim[0] - 100)
        self.trail_set = []

    def info(self, vx, vy, theta, v1,v2,v3):
        text = f"Vx ={np.round(vx,2)}, Vy ={np.round(vy,2)}, Theta ={np.round(theta,2)}, V1={np.round(v1,2)}, V2 ={np.round(v2,2)}, V3={np.round(v3,2)}"
        self.text1 = self.font.render(text, True, self.black, self.white)
        self.map.blit(self.text1, self.textRect)
    def sensor_info(self):
        text = f"sensor : {self.edge_distances}"
        self.text2 = self.font.render(text, True, self.black, self.white)
        self.textRect.center = (self.width - 700, self.height - 50)
        self.map.blit(self.text2, self.textRect)
        
    def trail(self, pos):
        for i in range(0, len(self.trail_set)-1):
            pygame.draw.line(self.map, self.red, (self.trail_set[i][0], self.trail_set[i][1]), (self.trail_set[i+1][0], self.trail_set[i+1][1]))
        if self.trail_set.__sizeof__() > 30000:
            self.trail_set.pop(0)
        self.trail_set.append(pos)   
     
    def robot_frame(self, pos, rotation):
        n = 80
        centerx, centery = pos
        x_axis = (centerx + n*math.cos(-rotation), centery + n*math.sin(-rotation))
        y_axis = (centerx + n*math.cos(-rotation+math.pi/2), centery + n*math.sin(-rotation+math.pi/2))  
        pygame.draw.line(self.map, self.blue, pos, x_axis, 3) 
        pygame.draw.line(self.map, self.green, pos, y_axis, 3) 

    def update_sensor_data(self, pos, r):
        angles = [-r, np.pi/3-r, 2*np.pi/3-r, np.pi-r, 4*np.pi/3-r, 5*np.pi/3-r]
        edge_points = []
        edge_distances = []

        for angle in angles:
            distance = 0
            #print(pos)
            edge_x, edge_y = (int(pos[0]), int(pos[1]))
            #print(edge_x,edge_y)
            while self.track_copy.get_at((edge_x, edge_y)) != WHITE_COLOR:
                edge_x = int(pos[0] + distance * math.cos(angle))
                edge_y = int(pos[1] + distance * math.sin(angle))
                distance += 1
            edge_points.append((edge_x, edge_y))
            edge_distances.append(distance)
        self.edge_distances = edge_distances
        for point in edge_points:
            pygame.draw.line(self.map,(0,255,0), pos, point)
            pygame.draw.circle(self.map, (0,255,0), point, 5)

class Robot:
    def __init__(self, startpos, Img, width):
        #self.m2p = 3779.52 #m to px
        self.w = width
        self.x = startpos[0]
        self.y = startpos[1]
        self.theta = 0
        self.vl = 0 #pixel/s
        self.v2 = 0
        self.v3 = 0 
        self.vx = 0
        self.vy = 0
        self.theta_dot =0
        self.vxg = 0
        self.vyg = 0
        self.theta_d =0
        #graphics
        self.img = pygame.image.load(Img).convert_alpha()
        self.img = pygame.transform.scale(self.img, (70, 70))
        self.rotated = self.img
        self.rect = self.rotated.get_rect(center=(self.x, self.y))
    
    def draw(self, map):
        map.blit(self.rotated, self.rect)

    def move(self, event = None):
        if event is not None:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a:
                    self.vxg -= 100
                elif event.key == pygame.K_d:
                    self.vxg += 100
                elif event.key == pygame.K_w:
                    self.vyg -= 100
                elif event.key == pygame.K_s:
                    self.vyg += 100
                elif event.key == pygame.K_q:
                    self.theta_d += 0.1
                elif event.key == pygame.K_e:
                    self.theta_d -= 0.1
        inv_R = np.array([[np.cos(self.theta), -np.sin(self.theta), 0],[np.sin(self.theta), np.cos(self.theta), 0],[0,0,1]])
        l = 10
        inv_J1 = np.array([[1/np.sqrt(3),0, -1/np.sqrt(3)],[-1/3, 2/3,-1/3],[-1/(3*l), -1/(3*l), -1/(3*l)]]) 
        #r=4 
        #J2 = np.array ([[r,0,0,0],[0,r,0,0],[0,0,r,0],[0,0,0,r]])
        r=3 
        J2 = np.array ([[r,0,0],[0,r,0],[0,0,r],])
        V = np.linalg.inv(J2) @ np.linalg.inv(inv_J1) @ np.linalg.inv(inv_R) @ np.array([[self.vxg],[self.vyg],[self.theta_d]])
        self.v1 = V[0,0]
        self.v2 = V[1,0]
        self.v3 = V[2,0]
        AAA = inv_R @ inv_J1 @ J2 @ np.array([[self.v1],[self.v2],[self.v3]])
        self.vx = AAA[0,0]
        self.vy = AAA[1,0]
        self.theta_dot = AAA[2,0]
        
        self.x =self.x + self.vx*dt
        self.y =self.y + self.vy*dt
        self.theta = self.theta + self.theta_dot*dt
        self.rotated = pygame.transform.rotozoom(self.img, math.degrees(self.theta), 1)
        self.rect = self.rotated.get_rect(center=(self.x, self.y))
        
    def crash_check(self):
        """
        Check if any corner of the car goes out of the track
        Returns:
            Bool: Returns True if the car is alive
        """
        for corner in self.corners:
            if self.img.get_at(corner) == WHITE_COLOR:
                return True
        return False
#initialisation
pygame.init()

start = (200, 300)
dims = (720, 1280)

running = True

dt = 0
lasttime = pygame.time.get_ticks()
environment = Envir(dims)
robot = Robot(start, "robot.png", 1)

#simulation loop
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        robot.move(event)

    dt = (pygame.time.get_ticks() - lasttime) / 1000
    lasttime = pygame.time.get_ticks()
    pygame.display.update()
    environment.map.blit(environment.track, (0, 0))
    #environment.write_info(int(robot.vl), int(robot.vr), robot.theta)
    robot.move()
    robot.draw(environment.map)
    environment.robot_frame((robot.x, robot.y), robot.theta)
    environment.update_sensor_data((robot.x, robot.y), robot.theta)
    environment.trail((robot.x, robot.y))
    environment.sensor_info()
  
    
