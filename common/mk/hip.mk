# (c) 2007 The Board of Trustees of the University of Illinois.

# Default language wide options

# HIP specific
LANG_CFLAGS=-I$(PARBOIL_ROOT)/common/include -I$(HIP_PATH)/include
LANG_CXXFLAGS=$(LANG_CFLAGS)
LANG_LDFLAGS=-L$(HIP_LIB_PATH)

LANG_HIPCFLAGS=$(LANG_CFLAGS)

CFLAGS=$(APP_CFLAGS) $(LANG_CFLAGS) $(PLATFORM_CFLAGS)
CXXFLAGS=$(APP_CXXFLAGS) $(LANG_CXXFLAGS) $(PLATFORM_CXXFLAGS)

HIPCFLAGS=$(LANG_HIPCFLAGS) $(PLATFORM_HIPCFLAGS) $(APP_HIPCFLAGS) 
HIPLDFLAGS=$(LANG_LDFLAGS) $(PLATFORM_HIPLDFLAGS) $(APP_HIPLDFLAGS)

# Rules common to all makefiles

########################################
# Functions
########################################

# Add BUILDDIR as a prefix to each element of $1
INBUILDDIR=$(addprefix $(BUILDDIR)/,$(1))

# Add SRCDIR as a prefix to each element of $1
INSRCDIR=$(addprefix $(SRCDIR)/,$(1))


########################################
# Environment variable check
########################################

# The second-last directory in the $(BUILDDIR) path
# must have the name "build".  This reduces the risk of terrible
# accidents if paths are not set up correctly.
ifeq ("$(notdir $(BUILDDIR))", "")
$(error $$BUILDDIR is not set correctly)
endif

ifneq ("$(notdir $(patsubst %/,%,$(dir $(BUILDDIR))))", "build")
$(error $$BUILDDIR is not set correctly)
endif

.PHONY: run

ifeq ($(HIP_PATH),)
FAILSAFE=no_hip
else 
FAILSAFE=
endif

########################################
# Derived variables
########################################

ifeq ($(DEBUGGER),)
DEBUGGER=gdb
endif

OBJS = $(call INBUILDDIR,$(SRCDIR_OBJS))

########################################
# Rules
########################################

default: $(FAILSAFE) $(BUILDDIR) $(BIN)

run:
	@echo "Resolving HIP runtime library..."
	#@$(shell echo $(RUNTIME_ENV)) LD_LIBRARY_PATH=$(HIP_LIB_PATH) ldd $(BIN) | grep hip
	@$(shell echo $(RUNTIME_ENV)) LD_LIBRARY_PATH=$(HIP_LIB_PATH) $(BIN) $(ARGS)

debug:
	@echo "Resolving HIP runtime library..."
	#@$(shell echo $(RUNTIME_ENV)) LD_LIBRARY_PATH=$(HIP_LIB_PATH) ldd $(BIN) | grep hip
	@$(shell echo $(RUNTIME_ENV)) LD_LIBRARY_PATH=$(HIP_LIB_PATH) $(DEBUGGER) --args $(BIN) $(ARGS)

clean :
	rm -rf $(BUILDDIR)/*
	if [ -d $(BUILDDIR) ]; then rmdir $(BUILDDIR); fi

$(BIN) : $(OBJS) $(BUILDDIR)/parboil_hip.o
	$(HIPLINK) $^ -o $@ $(HIPLDFLAGS)

$(BUILDDIR) :
	mkdir -p $(BUILDDIR)

$(BUILDDIR)/%.o : $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILDDIR)/parboil_hip.o: $(PARBOIL_ROOT)/common/src/parboil_hip.c
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILDDIR)/%.o : $(SRCDIR)/%.cc
	$(HIPCC) $(CXXFLAGS) -c $< -o $@

$(BUILDDIR)/%.o : $(SRCDIR)/%.cpp
	$(HIPCC) $(CXXFLAGS) -c $< -o $@

no_hip:
	@echo "HIP_PATH is not set. Open $(HIP_ROOT)/common/Makefile.conf to set default value."
	@echo "You may use $(PLATFORM_MK) if you want a platform specific configurations."
	@exit 1

