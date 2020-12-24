int main(){
    int i = 10;
    if(i < 5){
        for(;i < 10;i++){
            if(i < 3){
                if (i < 1){
                    return 0;
                }
                else{
                    return 2;
                }
            }
        }
    }
    while(i < 10){
        if(i == 10){
            break;
        }
        else
            i++;
    }
    return 0;
}