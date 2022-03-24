def divisors(num: int) -> list:
    """Found the divisors of num and returns it in a list"""
    div = []
    
    for i in range(1, num + 1):
        if num % i == 0:
            div.append(i)
     
    return div


def unique(l: list) -> list:
    """Returns a copy of l without repeating items"""
    copy = l.copy()
    
    for elem in l:
        
        if copy.count(elem) != 1:
            for i in range(copy.count(elem) - 1):#removes all but one repeating elements
                copy.remove(elem)
                
    return copy


def is_prime(num: int) -> bool:
    """Checks if num is a squarefree number and return true or false according to the case"""

    return len(divisors(num)) == 2 #a prime only has two divisors

def primes_up_to(num: int) -> list:
    """Returns a list with the prime numbers less or equals to num"""
    primes = []
    
    for i in range(2, num + 1):
        if is_prime(i):
            primes.append(i)
            
    return primes

