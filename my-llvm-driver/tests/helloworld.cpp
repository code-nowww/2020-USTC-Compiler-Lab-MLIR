int _fib(int x, int y){
    if(x == 0){
        return 1;
    }
    else if(x == 1){
        return 2;
    }
    else
    {
        return _fib(x - 1, 0) + _fib(x -2, 0);
    }
    
}

int main(){
    int fib = _fib(4, 0);
    for(int i = 0 ;i < 10; i++);
    return fib;
}