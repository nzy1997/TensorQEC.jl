n=2
xcodenum = 0
zcodenum = 0
matrix = falses(2*n^2-2, 4*n^2)
vertical_edges = reshape(1:n^2, n, n)
horizontal_edges = reshape(n^2+1:2*n^2, n, n)
for i in 0:n-1
    for j in 0:n-1
        if i ==n-1 && j == n-1
            break
        end
        xcodenum += 1
        matrix[xcodenum,horizontal_edges[i+1,j+1]] = true
        matrix[xcodenum,horizontal_edges[mod(i+1,n)+1,j+1]] = true
        matrix[xcodenum,vertical_edges[i+1,j+1]] = true
        matrix[xcodenum,vertical_edges[i+1,mod(j+1,n)+1]] = true
    end
end

for i in 0:n-1
    for j in 0:n-1
        if i ==n-1 && j == n-1
            break
        end
        zcodenum += 1
        matrix[zcodenum+xcodenum,vertical_edges[i+1,j+1]+2*n^2] = true
        matrix[zcodenum+xcodenum,vertical_edges[mod(i-1,n)+1,j+1]+2*n^2] = true
        matrix[zcodenum+xcodenum,horizontal_edges[i+1,j+1]+2*n^2] = true
        matrix[zcodenum+xcodenum,horizontal_edges[i+1,mod(j-1,n)+1]+2*n^2] = true
    end
end
