from .particle import Particle
import operator

operators = {
    'MIN': operator.lt,
    'MAX': operator.gt
}

class PSO:
    def __init__(self, func, size, beta, gamma, delta, alpha, epsilon, informantType, informantNB, maxIter, opt):
        self._betaCognitive = beta
        self._gammaSocial = gamma
        self._deltaGlobal = delta
        self._alphaIntertia = alpha
        self._epsilonStep = epsilon
        self._gBestParticle = None
        self._particles = []
        self._infortmantType = informantType
        self._informantNB = informantNB
        self._maxIter = maxIter
        self._opt = operators.get(opt)

        #create particles
        for i in range(size):
            particle = Particle(func)
            self._particles.append(particle)
        i=0
        #set informants for each particle
        for particle in self._particles:
            particles = self._particles.copy()
            del particles[i]
            particle.setInformants(informantType, informantNB, particles)
            i +=1
        
    
    def evolve(self):
        i=0
        while i < self._maxIter:
            j = 0
            self._gBestParticle = self._particles[0]
            for p in self._particles:
                # acc, loss =
                p.computeFitness(self._opt)
                if self._opt(p.getpBestFitness(), self._gBestParticle.getpBestFitness()):
                    self._gBestParticle = p
                
                #update informants => no need to update if random
                particles = self._particles.copy()
                del particles[j]

                p.updateInformants(self._infortmantType, self._informantNB, particles)
                j += 1
            
            #move all particles
            for p in self._particles:
                p.move(self._gBestParticle, self._betaCognitive, self._gammaSocial, self._deltaGlobal, self._alphaIntertia, self._epsilonStep, self._opt)

            i += 1

            #progress 
            if i % (self._maxIter/10) ==0:
                print(self._gBestParticle.getpBestFitness())
                print('.')

        self._gBestParticle.updateFunc()
        return self._gBestParticle.getFunc()

