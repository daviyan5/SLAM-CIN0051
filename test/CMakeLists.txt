add_subdirectory(frontend)
add_subdirectory(preprocessing)
add_subdirectory(backend)

install(
    DIRECTORY data
    DESTINATION test
)

install(
    DIRECTORY
    DESTINATION test/results
)
