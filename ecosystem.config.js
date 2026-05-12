module.exports = {
  apps: [
    {
      name: "proxy-vertex",
      script: "start.sh",
      cwd: "/root/proxy-vertex-ai-google-console",
      interpreter: "bash",
      autorestart: true,
      watch: false,
      max_memory_restart: "300M",
    },
  ],
};
