% Sample protein sequence
% Specify the path to your FASTA file
fasta_file = 'P37840.fasta.txt'; %alpha-synuclein
% fasta_file = 'P10636-8.fasta.txt'; %human tau protein
% fasta_file = 'P01308.fasta.txt'; %insulin


% Read the FASTA file
fasta_data = fastaread(fasta_file);

% Extract the protein sequence from the FASTA data
protein_sequence = fasta_data.Sequence;

% Define the amino acid residues
residues = 'ACDEFGHIKLMNPQRSTVWY';
residues_cell =  num2cell(residues);

% Initialize an array to store the count of each residue
residue_count = zeros(1, length(residues));

for residue_idx = length (residues)
    residue_find = residues(residue_idx);

    residue_positions = {};

% Loop through the protein sequence and count the occurrences of each residue
for i = 1:length(protein_sequence)
    residue_index = strfind(residues, protein_sequence(i));
    residue_count(residue_index) = residue_count(residue_index) + 1;
    
    if protein_sequence (i) == residue_find
    residue_positions = {residue_positions i}; 
    end
end
end


    
% Display the results
% disp('Residue   Count');
% disp('-------   -----');
% for i = 1:length(residues)
%     fprintf('%6s     %d\n', residues(i), residue_count(i));
% end

% Plot the results
figure;
bar(categorical(residues_cell), residue_count);
ylabel('counts');
