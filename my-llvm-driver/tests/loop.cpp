// void test() {
//     int num = 0;
//     L2:
//         for(int i=0;i<10;i++){
//             for(int j=i;j<10;j++){
//                 num++;
//             }
//         }
//         while (num < 10) {
//             num++;
//         }
//     goto L2;
// }

int main(){
    int num = 0;
    for(int i=0;i<10;i++){
        for(int j=i;j<10;j++){
            for(int k=j;k<10;k++){
				num++;
            }
            for(int k=j;k<10;k++){
				num++;
            }
        }
        for(int j=i;j<10;j++){
            for(int k=j;k<10;k++){
				num++;
            }
        }
    }
    L1:
        for(int i=0;i<10;i++){
            for(int j=i;j<10;j++){
                num++;
            }
        }
        while (num < 10) {
            num++;
        }
    goto L1;
    return num;
}