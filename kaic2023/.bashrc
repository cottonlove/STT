# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=



if [[ $- != *i* ]] ; then
  # shell is non-interactive - do a silent return
  return
fi

# User specific aliases and functions

echo ""
echo ""
echo " -*-*-*-*-*-Welcome to NOVA Platform-*-*-*-*-*-*-"
echo " ________   ________  ___      ___ ________      "
echo "|\   ___  \|\   __  \|\  \    /  /|\   __  \     "
echo " \ \  \\ \  \ \  \|\  \ \  \  /  / | \  \|\  \    "
echo "  \ \  \\ \  \ \  \ \\  \ \  \/  / / \ \   __  \   "
echo "   \ \  \\ \  \ \  \ \\  \ \    / /   \ \  \ \  \  "
echo "    \ \__\\ \__\ \_______\ \__/ /     \ \__\ \__\ "
echo "     \|__|\|__|\|_______|\|__|/       \|__|\|__| "
echo "                                                   "




docker exec -it kaic /bin/bash &&cd ~

