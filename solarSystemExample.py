from logic import *

#Converting first paragraph of text from this website for children into first order logic form: http://www.kidsastronomy.com/solar_system.htm

#Initialize solar system database. Note that model checking KB seems to get too slow as database gets larger
#However resolution KB stops working whenever there are more than just a few variables.
kb = createResolutionKB()

print "The Solar System is made up of all the planets that orbit our Sun."
solarSystem = Constant('solar system')
sun = Constant('sun')
earth = Constant('earth')
def contain(x,y): return Atom("Contains", x, y) #X made up of Y
def planet(x): return Atom("Planet", x)
def orbit(x,y): return Atom('Orbit', x, y) #X orbits around Y
kb.tell(Forall('$x', Implies(And(orbit('$x', sun), planet('$x')), contain(solarSystem, '$x'))))
kb.tell(planet(earth))
kb.tell(orbit(earth, sun))
print " - Does the solar system contain the earth?", kb.ask(contain(solarSystem, earth))

print "In addition to planets, the Solar System also consists of moons, comets, asteroids, minor planets, and dust and gas."
def moon(x): return Atom("Moon", x)
def comet(x): return Atom("Comet", x)
def asteroid(x): return Atom("Asteroid", x)
def minorPlanet(x): return Atom("Minor planet", x)
def dust(x): return Atom("Dust", x)
def gas(x): return Atom("Gas", x)
kb.tell(Exists('$x', And(contain(solarSystem, '$x'), moon('$x'))))
kb.tell(Exists('$x', And(contain(solarSystem, '$x'), comet('$x'))))
kb.tell(Exists('$x', And(contain(solarSystem, '$x'), asteroid('$x'))))
kb.tell(Exists('$x', And(contain(solarSystem, '$x'), minorPlanet('$x'))))
kb.tell(Exists('$x', And(contain(solarSystem, '$x'), dust('$x'))))
kb.tell(Exists('$x', And(contain(solarSystem, '$x'), gas('$x'))))

print "Everything in the Solar System orbits or revolves around the Sun."
def revolveAround(x,y): return Atom("Revolve around", x, y)
kb.tell(Forall('$x', Implies(contain(solarSystem, '$x'), Or(orbit('$x', sun), revolveAround('$x', sun)))))

print "The Sun contains around 98% of all the material in the Solar System."
def larger(x,y): return Atom("Larger", x, y) # x larger than y
print kb.tell(Forall('$x', Implies(contain(solarSystem, '$x'), larger(sun, '$x'))))

print "The larger an object is, the more gravity it has."
print kb.tell(Forall('$x', Forall('$y', Not(And(larger('$x', '$y'), larger('$y', '$x'))))))

# #These statements lead to extremely slow performance
def gravity(x): return Atom("Gravity", x)
def has(x,y): return Atom("Has", x, y)
# print kb.tell(Forall('$x', Forall('$y', Implies(larger('$x', '$y'), Forall('$g1', Forall('$g2', Implies(And(
#     And(gravity('$g1'), has('$x','$g1')),
#     And(gravity('$g2'), has('$y','$g2'))), larger('$g1', '$g2'))))))))
# print kb.tell(Forall('$z', Forall('$y', Forall('$x', Implies(And(larger('$x', '$y'), larger('$y', '$z')), larger('$x', '$z'))))))

print "Because the Sun is so large, its powerful gravity attracts all the other objects in the Solar System towards it."
def attracts(x, y): return Atom("Attracts", x, y) #X attracts Y
kb.tell(Forall('$x', Forall('$y', Implies(larger('$x','$y'), attracts('$x', '$y')))))
kb.tell(Forall('$x', Forall('$y', Implies(attracts('$x', '$y'), Not(attracts('$y', '$x'))))))
print " - Does the sun attract every object in the solar system?", kb.ask(Forall('$x', Implies(contain(solarSystem, '$x'), attracts(sun, '$x'))))
print " - Is there any minor planet in the solar system that attracts the sun towards itself?", kb.ask(Exists('$y', AndList([minorPlanet('$y'), contain(solarSystem, '$y'), attracts('$y', sun)])))

print "At the same time, these objects, which are moving very rapidly, try to fly away from the Sun, outward into the emptiness of outer space."
def movingRapidly(x): return Atom("Moving Rapidly", x)
def moveAway(x,y): return Atom("Moving away from", x, y) #x moving away from y
def moveTowards(x,y): return Atom("moving towards", x, y)
outerSpace = Constant('outer space')

print "The result of the planets trying to fly away, at the same time that the Sun is trying to pull them inward is that they become trapped half-way in between."

print "Balanced between flying towards the Sun, and escaping into space, they spend eternity orbiting around their parent star."