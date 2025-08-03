add_subdirectory(preprocessing)

install(
    DIRECTORY data
    DESTINATION test
)

install(
    DIRECTORY
    DESTINATION test/results
)
