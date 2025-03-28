#!/bin/bash

# Remove existing projects
rm -rf project_4*

# Array of sparsity values
sparsity_values=(1)

# Loop through each sparsity value
for sparsity in "${sparsity_values[@]}"; do

    # Run make and vitis_hls

    make SPARSE_LEVEL="-DSPARSE_${sparsity}"
    vitis_hls -f script_${sparsity}.tcl

    # Move the generated project to a new directory
    new_dir="project_4_$(echo $sparsity | tr '.' '_')"
    mv project_4/ "$new_dir"

    echo "Moved project to $new_dir"
done

echo "All runs completed."
