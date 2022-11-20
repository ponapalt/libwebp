call vcvars32
git fetch origin
git reset --hard origin/main
nmake /f makefile.vc CFG=release-static OBJDIR=.\obj RTLIBCFG=legacy clean
nmake /f makefile.vc CFG=release-static OBJDIR=.\obj RTLIBCFG=legacy
