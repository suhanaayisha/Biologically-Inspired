import sys
import numpy as np
import secrets
from random import sample

class Particle:
    def __init__(self, func):
        self._func = func
        self.velocity = 0
        self.position,  self.pBestPos = None, None
        self.pBestFitness = 0
        self.fitness = 0
        self._informants = []
        dimensions = func.getDimension()
        self.initPosition(dimensions)
    
    def initPosition(self, dim):
        self.position = np.random.uniform(-1,1,dim)
        self.pBestPos = np.copy(self.position)

    def getPosition(self):
        return self.position
    
    def distance(self,p):
        dist = np.linalg.norm(self.position - p.getPosition())
        return dist

    def findNeighbours(self, particles, informantNB):
        neighbours = []
        for particle in particles:
            distance = self.distance(particle)
            neighbours.append((particle,distance))
        neighbours.sort(key=lambda tup: tup[1])
        sortedParticles = []
        for neighbour in neighbours:
            particle = neighbour[0]
            sortedParticles.append(particle)
        return sortedParticles[0:int(informantNB)]

        return sortedParticles[0:informantNB]

    def setInformants(self, informantType, informantNB, particles):
        if informantType == 0:
            self._informants = sample(particles, informantNB)
        elif informantType == 1:
            self._informants = self.findNeighbours(particles, informantNB)
        else:
            self._informants = self.findNeighbours(particles, int(informantNB/2))
            del particles[0:int(informantNB/2)]
            self._informants.extend(sample(particles, int(informantNB/2)))

    def updateInformants(self, informantType, informantNB, particles):
        if informantType == 1:
            self._informants = self.findNeighbours(particles, informantNB)
        elif informantType == 2:
            self._informants = self.findNeighbours(particles, int(informantNB/2))
            del particles[0:int(informantNB/2)]
            self._informants.extend(sample(particles, int(informantNB/2)))

    def getpBestFitness(self):
        return self.pBestFitness
    
    def getpBestPosition(self):
        return self.pBestPos

    def computeFitness(self, opt):
        acc, loss = self._func.evaluate(self.position)
        if opt(loss, self.pBestFitness):
            self.pBestPos = np.copy(self.position)
            self.pBestFitness = loss
        return acc, loss

    def updateFunc(self):
        self._func.setVariables(self.pBestPos)

    def findlBestParticle(self, opt):
        lBestParticle = self._informants[0]
        for informant in self._informants:
            if opt(informant.getpBestFitness(), lBestParticle.getpBestFitness()):
                lBestParticle = informant
        return lBestParticle

    def getFunc(self):
        return self._func
    
    def move(self, gBest, beta, gamma, delta, alpha, epsilon, opt):
        localBest = self.findlBestParticle(opt)

        seed = secrets.randbits(128)
        rng = np.random.default_rng(seed)
        self.velocity = alpha * self.velocity
        self.velocity += beta *  rng.random(size=self.position.shape[0]) * (self.pBestPos - self.position)
        self.velocity += gamma *  rng.random(size=self.position.shape[0]) * (localBest.getpBestPosition() - self.position)
        self.velocity += delta *  rng.random(size=self.position.shape[0]) * (gBest.getpBestPosition() - self.position)

        self.position =  self.position + epsilon +  self.velocity

        return self.pBestPos, self.pBestFitness
        
    
        