for x in $@; do
    jupyter nbconvert --to notebook --inplace --ExecutePreprocessor.kernel_name=python3 --execute $x
done
