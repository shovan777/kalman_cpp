set(srcs "hello_world_main.cpp"
            "matops.cpp")
idf_component_register(SRCS "${srcs}"
                    INCLUDE_DIRS ".")