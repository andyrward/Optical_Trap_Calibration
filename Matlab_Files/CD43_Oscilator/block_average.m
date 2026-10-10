function [block_avg, block_std] = block_average(arr, Nb)
    % Ensure the length of the array is a multiple of Nb
    N = length(arr);
    
    % If the array length is not a multiple of Nb, truncate the extra elements
    if mod(N, Nb) ~= 0
        arr = arr(1:floor(N/Nb)*Nb);
    end
    
    % Reshape the array into a matrix where each row is a block of size Nb
    arr_reshaped = reshape(arr, Nb, []);
    
    % Compute the mean along each column (i.e., for each block)
    block_avg = mean(arr_reshaped, 1);
    
    % Compute the standard deviation along each column (i.e., for each block)
    block_std = std(arr_reshaped, 0, 1);
    
    % Return as row vectors
    block_avg = block_avg(:)';
    block_std = block_std(:)';
end
