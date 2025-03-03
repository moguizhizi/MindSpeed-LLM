# This script is used to run the coverage test for the pipeline, unit tests and shell scripts.
# define the base directory
BASE_DIR=$(dirname $(dirname "$(readlink -f "$0")"))
SOURCE_DIR="$BASE_DIR/mindspeed_llm"
PIPELINE_DIR="$BASE_DIR/tests/pipeline"
UT_DIR="$BASE_DIR/tests/ut"
ST_DIR="$BASE_DIR/tests/st/shell_scripts"

# remove the existing coverage files
rm -f .coverage
rm -f .coverage.*
rm -rf htmlcov


# create the coverage configuration file
cat > ".coveragerc" << EOF
[run]
branch = False
parallel = False
source = $SOURCE_DIR

[report]
show_missing = True
skip_covered = False
EOF


add_coverage() {
    sed -i "1a\import random" pretrain_gpt.py
    sed -i "2a\import time" pretrain_gpt.py
    sed -i "3a\import coverage" pretrain_gpt.py
    sed -i '4a\cov = coverage.Coverage(data_suffix=f"usecase-{time.time_ns()}_{random.randint(0, 100)}")' pretrain_gpt.py
    sed -i "5a\cov.start()" pretrain_gpt.py

    sed -i "/    main()/a\    cov.stop()" pretrain_gpt.py
    sed -i "/    cov.stop()/a\    cov.save()" pretrain_gpt.py

    sed -i "1a\import random" posttrain_gpt.py
    sed -i "2a\import time" posttrain_gpt.py
    sed -i "3a\import coverage" posttrain_gpt.py
    sed -i '4a\cov = coverage.Coverage(data_suffix=f"usecase-{time.time_ns()}_{random.randint(0, 100)}")' posttrain_gpt.py
    sed -i "5a\cov.start()" posttrain_gpt.py

    sed -i "/    launch()/a\    cov.stop()" posttrain_gpt.py
    sed -i "/    cov.stop()/a\    cov.save()" posttrain_gpt.py

    sed -i "1a\import random" ray_gpt.py
    sed -i "2a\import time" ray_gpt.py
    sed -i "3a\import coverage" ray_gpt.py
    sed -i '4a\cov = coverage.Coverage(data_suffix=f"usecase-{time.time_ns()}_{random.randint(0, 100)}")' ray_gpt.py
    sed -i "5a\cov.start()" ray_gpt.py

    sed -i "/    main()/a\    cov.stop()" ray_gpt.py
    sed -i "/    cov.stop()/a\    cov.save()" ray_gpt.py
}


remove_coverage() {
    sed -i "2d" pretrain_gpt.py
    sed -i "2d" pretrain_gpt.py
    sed -i "2d" pretrain_gpt.py
    sed -i "2d" pretrain_gpt.py
    sed -i "2d" pretrain_gpt.py

    sed -i "/    cov.stop()/d" pretrain_gpt.py
    sed -i "/    cov.save()/d" pretrain_gpt.py

    sed -i "2d" posttrain_gpt.py
    sed -i "2d" posttrain_gpt.py
    sed -i "2d" posttrain_gpt.py
    sed -i "2d" posttrain_gpt.py
    sed -i "2d" posttrain_gpt.py

    sed -i "/    cov.stop()/d" posttrain_gpt.py
    sed -i "/    cov.save()/d" posttrain_gpt.py

    sed -i "2d" ray_gpt.py
    sed -i "2d" ray_gpt.py
    sed -i "2d" ray_gpt.py
    sed -i "2d" ray_gpt.py
    sed -i "2d" ray_gpt.py

    sed -i "/    cov.stop()/d" ray_gpt.py
    sed -i "/    cov.save()/d" ray_gpt.py
}


# run the coverage for python files in the pipeline
find "$PIPELINE_DIR" -mindepth 1 -maxdepth 1 -type d | while read -r dir; do
    if [ -d "$dir" ]; then
        find "$dir" -type f -name "*.py" | while read -r file; do
            coverage run -p --source=$SOURCE_DIR $file
        done
    fi
done

# run the coverage for python files in the unit tests
find "$UT_DIR" -mindepth 1 -maxdepth 1 -type d | while read -r dir; do
    if [ -d "$dir" ]; then
        find "$dir" -type f -name "*.py" | while read -r file; do
            coverage run -p --source=$SOURCE_DIR $file
        done
    fi
done

add_coverage

# run the coverage for shell scripts in the st
for test_case in "$ST_DIR"/*.sh; do
    file_name=$(basename "${test_case}")
    echo "Running $file_name..."
    bash $test_case
    SCRIPT_EXITCODE=${PIPESTATUS[0]}
    if [ $SCRIPT_EXITCODE -ne 0 ]; then
        echo "Script has failed. Exit!"
        remove_coverage()
        exit 1
    fi
done

# run the coverage for shell scripts in the pipeline
find "$PIPELINE_DIR" -mindepth 1 -maxdepth 1 -type d | while read -r dir; do
    if [ -d "$dir" ]; then
        find "$dir" -type f -name "*.sh" | while read -r file; do
            bash $file
            SCRIPT_EXITCODE=${PIPESTATUS[0]}
            if [ $SCRIPT_EXITCODE -ne 0 ]; then
                echo "Script has failed. Exit!"
                remove_coverage()
                exit 1
            fi
        done
    fi
done

remove_coverage


# generate the coverage report
coverage combine
coverage html
