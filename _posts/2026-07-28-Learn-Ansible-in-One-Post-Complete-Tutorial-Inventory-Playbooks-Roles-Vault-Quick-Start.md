---
layout: post
title: "Learn Ansible in a Single Post: A Complete Tutorial From Inventory and Playbooks to Roles, Vault, and Production"
description: "A complete Ansible tutorial in one blog post. Covers the whole tool in 5 stages: inventory (hosts, groups, dynamic inventory, variables), playbooks (plays, tasks, YAML, idempotency, modules apt/copy/template/service/git, handlers), roles (directory structure, ansible-galaxy, collections, variable precedence), and production (ansible-vault for secrets, ansible-lint, Molecule testing, AWX/Tower automation, pull mode, fact caching). Five hand-drawn diagrams, runnable YAML, and a quick-start roadmap."
date: 2026-07-28
header-img: "img/post-bg.jpg"
permalink: /Learn-Ansible-in-One-Post-Complete-Tutorial-Inventory-Playbooks-Roles-Vault-Quick-Start/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - Ansible
  - Configuration Management
  - DevOps
  - IaC
  - Automation
  - Tutorial
categories: [Tutorial, DevOps, Infrastructure]
keywords: "Ansible tutorial one post, learn Ansible fast, Ansible inventory hosts groups dynamic variables, Ansible playbook plays tasks YAML idempotency, Ansible modules apt copy template service git, handlers notify, Ansible roles directory structure ansible-galaxy collections, variable precedence defaults inventory group host play extra vars, ansible-vault encrypt secrets, ansible-lint Molecule testing, AWX Ansible Tower automation, ansible-pull mode, fact caching, Ansible vs Terraform, Ansible quick start roadmap"
author: "PyShine"
image: /assets/img/diagrams/ansible-tutorial/ans-flow.svg
---

# Learn Ansible in a Single Post: Complete Tutorial From Inventory and Playbooks to Roles, Vault, and Production

Ansible is an **agentless** configuration-management tool — you write YAML "playbooks" that describe the desired state of your servers, and Ansible uses SSH to make them match. No agent installed on the managed hosts; no master server; idempotent (running the same playbook twice produces the same result). It's the tool for provisioning servers, deploying apps, and managing configuration at scale, alongside [Terraform](/Learn-Terraform-in-One-Post-Complete-Tutorial-HCL-State-Modules-Providers-Quick-Start/) (which provisions the infrastructure itself). This single post covers the whole tool in five stages, with hand-drawn diagrams and runnable YAML.

## Learning Roadmap

![Ansible Roadmap](/assets/img/diagrams/ansible-tutorial/ans-roadmap.svg)

The roadmap moves from inventory (Stage 1), through playbooks (Stage 2), modules (Stage 3), roles (Stage 4), and production (Stage 5). The [Linux CLI tutorial](/Learn-Linux-CLI-in-One-Post-Complete-Tutorial-Files-Processes-Text-Quick-Start/) is a prerequisite — Ansible runs Linux commands.

---

## Stage 1 — Inventory

### The control node and the managed hosts

![Ansible Flow: Control Node -> SSH -> Hosts](/assets/img/diagrams/ansible-tutorial/ans-flow.svg)

Ansible runs on a **control node** (your laptop or a CI server). It connects to **managed hosts** over SSH (or WinRM for Windows). **No agent is installed on the managed hosts** — Ansible pushes Python modules over SSH, runs them, and removes them. This is the key difference from Chef/Puppet/Salt (which install an agent).

### The inventory file

The inventory lists the hosts Ansible manages, organized into groups:

```ini
# inventory.ini
[webservers]
web1.example.com
web2.example.com ansible_user=ubuntu

[databases]
db1.example.com ansible_port=2222

[production:children]
webservers
databases

[all:vars]
ansible_user=admin
ansible_ssh_private_key_file=~/.ssh/id_rsa
```

- **Groups** (`[webservers]`, `[databases]`) — organize hosts by role.
- **Group of groups** (`[production:children]`) — nest groups.
- **Group variables** (`[all:vars]`) — variables applied to all hosts in the group.
- **Host-specific variables** (`ansible_user=ubuntu` on a host line) — override per host.

### Dynamic inventory

For cloud environments, hosts come and go. **Dynamic inventory** queries the cloud API (AWS, GCP, Azure) at runtime to get the current host list:

```bash
# AWS dynamic inventory plugin
ansible-inventory --list -i aws_ec2.yaml
```

```yaml
# aws_ec2.yaml
plugin: aws_ec2
regions:
  - us-east-1
keyed_groups:
  - key: tags.Environment   # group hosts by their Environment tag
```

---

## Stage 2 — Playbooks

### A playbook = plays → tasks → modules

A **playbook** is a list of **plays**. A play targets a set of hosts and runs a list of **tasks**. Each task calls a **module** (a unit of work: install a package, copy a file, start a service).

![Playbook: Plays -> Tasks -> Modules](/assets/img/diagrams/ansible-tutorial/ans-playbook.svg)

```yaml
# playbook.yml
- name: Configure webservers
  hosts: webservers
  become: yes              # run as root (sudo)
  tasks:
    - name: Install nginx
      apt:
        name: nginx
        state: present
        update_cache: yes

    - name: Copy nginx config
      template:
        src: nginx.conf.j2
        dest: /etc/nginx/nginx.conf
      notify: restart nginx

    - name: Ensure nginx is running
      service:
        name: nginx
        state: started
        enabled: yes

  handlers:
    - name: restart nginx
      service:
        name: nginx
        state: restarted
```

```bash
ansible-playbook -i inventory.ini playbook.yml
```

### Idempotency — the core idea

Running this playbook once installs nginx, copies the config, and starts the service. Running it **again** does nothing — `apt: state: present` sees nginx is already installed and skips; `service: state: started` sees it's running and skips. **Only the `template` task runs every time** (it checks if the file content differs), and only if the config *changed* does it `notify` the handler to restart nginx.

This is idempotency: the playbook describes the **desired state**, and Ansible only makes changes needed to reach it. You can run it 100 times and the system ends up in the same state as running it once.

### Handlers — run only on change

A **handler** is a task that runs only when **notified** by another task that changed something. `notify: restart nginx` queues the handler; it runs at the end of the play, and only if at least one task notified it. This avoids restarting nginx when nothing changed (idempotency), and batches multiple notifications into one restart.

> **Pitfall:** A task without `state:` or with the wrong state isn't idempotent. `apt: name=nginx` (no `state`) defaults to `present` (idempotent), but `shell: echo hello > /tmp/file` runs *every* time (the shell module doesn't check state). Use modules (`copy`, `template`, `file`) over `shell`/`command` whenever possible — modules are idempotent; raw shell is not.

---

## Stage 3 — Modules

### Common modules

| Module | What it does |
|---|---|
| **`apt`** / **`yum`** / **`package`** | install/remove packages |
| **`copy`** | copy a file to the host |
| **`template`** | render a Jinja2 template and copy it |
| **`service`** / **`systemd`** | start/stop/enable a service |
| **`file`** | manage files/dirs/symlinks/permissions |
| **`user`** / **`group`** | manage users and groups |
| **`git`** | clone/pull a git repo |
| **`lineinfile`** | edit a single line in a file (idempotent) |
| **`blockinfile`** | edit a block of lines (idempotent) |
| **`shell`** / **`command`** | run a shell command (NOT idempotent by default) |
| **`debug`** | print a message (for debugging) |
| **`stat`** | check if a file exists |

### Templates (Jinja2)

```jinja2
{# nginx.conf.j2 #}
server {
    listen 80;
    server_name {{ nginx_server_name }};
    root {{ nginx_root }};
}
```

```yaml
# in the playbook
- template:
    src: nginx.conf.j2
    dest: /etc/nginx/nginx.conf
  vars:
    nginx_server_name: example.com
    nginx_root: /var/www/site
```

The `template` module renders the Jinja2 template (substituting `{{ variables }}`) and copies the result to the host — idempotently (only re-copies if the rendered content differs).

> **Pitfall:** `shell` and `command` modules are **not idempotent** — they run every time. If you must use them, guard with `creates:`/`removes:` (only run if a file exists/doesn't) or `changed_when:` (explicitly control the "changed" status). Prefer the dedicated modules (`file`, `lineinfile`, `copy`) which handle idempotency for you.

---

## Stage 4 — Roles

### The role directory structure

A **role** is a reusable, self-contained unit of configuration — the Ansible equivalent of a function or a library. Roles have a fixed directory structure:

![Roles: Structure, Galaxy, Collections](/assets/img/diagrams/ansible-tutorial/ans-roles.svg)

```
roles/
  nginx/
    tasks/main.yml        # the tasks to run
    handlers/main.yml     # handlers (notified by tasks)
    templates/            # Jinja2 templates
    files/                # static files to copy
    vars/main.yml         # role variables (high precedence)
    defaults/main.yml     # default variable values (lowest precedence)
    meta/main.yml         # role metadata + dependencies
```

```yaml
# playbook using a role
- hosts: webservers
  roles:
    - nginx
    - { role: postgres, when: "'databases' in group_names" }
```

### ansible-galaxy + collections

**ansible-galaxy** is the hub for community roles. Install a role:

```bash
ansible-galaxy install geerlingguy.nginx
```

**Collections** bundle multiple roles + plugins + modules (the modern distribution unit):

```bash
ansible-galaxy collection install community.general
```

Pin versions in `requirements.yml`:

```yaml
# requirements.yml
roles:
  - name: geerlingguy.nginx
    version: 3.1.0
collections:
  - name: community.general
    version: 4.8.0
```

```bash
ansible-galaxy install -r requirements.yml
```

### Variable precedence (highest to lowest)

```
extra vars (ansible-playbook -e)         ← highest
task vars
block vars
role + include vars
play vars
host vars
group vars
inventory vars
role defaults                            ← lowest
```

This is the source of most Ansible confusion — a variable set in multiple places resolves by precedence. `role defaults` are meant to be overridable; `extra vars` (`-e`) always win. When a variable "isn't taking," trace it through this list.

> **Pitfall:** Setting a variable in `group_vars/all.yml` and also in `host_vars/web1.yml` — the host var wins (higher precedence). If you expected the group var to apply to web1, you've shadowed it. Know the precedence order; use `role defaults` for overridable defaults.

---

## Stage 5 — Production

![Production: Vault, Lint, AWX, Molecule](/assets/img/diagrams/ansible-tutorial/ans-prod.svg)

### ansible-vault — encrypt secrets

```bash
# create an encrypted vars file
ansible-vault create group_vars/all/vault.yml
# edit it
ansible-vault edit group_vars/all/vault.yml
# run a playbook that uses the encrypted vars
ansible-playbook playbook.yml --ask-vault-pass
```

`ansible-vault` encrypts files with AES-256 — store secrets (DB passwords, API keys) in encrypted `vault.yml` files, commit them to Git (safe because encrypted), and decrypt at runtime with a password. Use a **vault password file** (or a password manager lookup) for CI.

### ansible-lint + Molecule

```bash
ansible-lint playbook.yml        # lint YAML + best practices
molecule test                    # test a role against a container (Testinfra)
```

- **ansible-lint** — catches common mistakes (unquoted YAML, missing `name:` on tasks, non-idempotent patterns).
- **Molecule** — spins up a container (Docker), runs the role, runs tests (Testinfra), destroys the container. The unit-test framework for Ansible roles.

### Dry-run + diff

```bash
ansible-playbook playbook.yml --check    # dry run: report what WOULD change, don't change
ansible-playbook playbook.yml --diff     # show the diff of every file change
ansible-playbook playbook.yml --check --diff   # both: see exactly what would change, safely
```

Always `--check --diff` before running against production — see what would change before you change it.

### AWX / Ansible Automation Platform

**AWX** is the open-source web UI for Ansible (the upstream of Red Hat's paid **Ansible Automation Platform** / "Tower"). It schedules playbook runs, manages inventories, provides RBAC, and surfaces run logs in a dashboard. For team use, AWX replaces "everyone runs ansible from their laptop."

### ansible-pull — hosts pull their own config

Normally Ansible **pushes** from the control node. `ansible-pull` reverses it — each host pulls its playbook from a Git repo and runs it locally (via cron). This is the "agentless agent" pattern for fleets where push is impractical (laptops, ephemeral autoscaled instances behind a NAT).

### Fact caching

Ansible **facts** (host details: OS, IP, disk) are gathered on every run, which is slow. **Fact caching** (in Redis, a file, or a DB) stores facts between runs, skipping re-gathering:

```yaml
# ansible.cfg
[defaults]
fact_caching = redis
fact_caching_connection = localhost:6379
```

### Ansible vs Terraform

| | Ansible | Terraform |
|---|---|---|
| **What it provisions** | config *inside* a server (packages, files, services) | the server *itself* (VMs, networks, load balancers) |
| **State** | stateless (checks current state each run) | stateful (state file tracks resources) |
| **Idempotent** | ✅ | ✅ |
| **Style** | imperative (you say how) | declarative (you say what) |

Common pattern: **Terraform creates the VMs + networking; Ansible configures the software on them.** Terraform for infra, Ansible for config.

---

## Quick-Start Checklist

1. **Install Ansible** — `pip install ansible` (or `apt install ansible`).
2. **Add hosts to `inventory.ini`** — `web1 ansible_host=1.2.3.4 ansible_user=ubuntu`.
3. **Test connectivity** — `ansible all -i inventory.ini -m ping`.
4. **Write a playbook** — one play, one task (`apt: name=nginx state=present`).
5. **Run it** — `ansible-playbook -i inventory.ini playbook.yml`.
6. **Run it again** — see "changed=0" (idempotency in action).
7. **Add a `template` task + a `handler`** — copy + restart on change.
8. **Create a role** — `ansible-galaxy init nginx` (the directory structure).
9. **Encrypt secrets** — `ansible-vault create group_vars/all/vault.yml`.
10. **Lint + dry-run** — `ansible-lint playbook.yml` + `--check --diff` before prod.

## Common Pitfalls

- **Non-idempotent `shell`/`command` tasks** — they run every time. Use modules or guard with `creates:`/`changed_when:`.
- **Missing `become: yes`** — tasks that need root fail with permission errors; add `become` at the play level.
- **No `--check --diff` before prod** — you didn't see what would change. Always dry-run.
- **Variable precedence confusion** — a var set in multiple places resolves by precedence; trace it.
- **Not pinning role versions** — `ansible-galaxy install` without a version gets the latest (could break). Pin in `requirements.yml`.
- **Secrets in plaintext** — commit `group_vars/all/vault.yml` *unencrypted* to Git. Use `ansible-vault`.
- **Forgetting `notify` only triggers on change** — a handler without a notifying task never runs.
- **Re-gathering facts every run** — slow; enable fact caching.

## Further Reading

- [Ansible Docs](https://docs.ansible.com/ansible/latest/) — the official reference
- [Ansible Galaxy](https://galaxy.ansible.com/) — community roles + collections
- [Ansible for DevOps](https://www.ansiblefordevops.com/) by Jeff Geerling — the canonical book
- [Molecule Docs](https://molecule.readthedocs.io/) — role testing
- [ansible-lint](https://ansible-lint.readthedocs.io/) — the linter

## Related guides

Ansible is the config-management layer of the DevOps stack — these PyShine tutorials connect to it:

- **[Learn Terraform in One Post](/Learn-Terraform-in-One-Post-Complete-Tutorial-HCL-State-Modules-Providers-Quick-Start/)** — the comparison; Terraform provisions infra, Ansible configures inside it.
- **[Learn Docker in One Post](/Learn-Docker-in-One-Post-Complete-Tutorial-Dockerfile-Volumes-Compose-Quick-Start/)** — Docker replaces a lot of config-management (the image IS the config); Ansible still provisions the host.
- **[Learn Kubernetes in One Post](/Learn-Kubernetes-in-One-Post-Complete-Tutorial-Pods-Deployments-Services-Production-Quick-Start/)** — K8s replaces config-management for containerized apps; Ansible for the K8s nodes.
- **[Learn Linux CLI in One Post](/Learn-Linux-CLI-in-One-Post-Complete-Tutorial-Files-Processes-Permissions-Quick-Start/)** — Ansible runs Linux commands; know the CLI first.
- **[Learn Git in One Post](/Learn-Git-in-One-Post-Complete-Tutorial-Branches-Rebase-Workflows-Quick-Start/)** — playbooks + roles live in Git; `requirements.yml` pins versions.

---

Ansible's value is **agentless, idempotent, YAML-driven configuration management** — describe the desired state, let Ansible converge to it, run it 100 times safely. The five stages here — inventory, playbooks, modules, roles, production — cover everything from a one-host `ping` to a vault-encrypted, Molecule-tested, AWX-scheduled, GitOps-driven production fleet with fact caching and dynamic cloud inventory. The two habits that pay off: **prefer modules over `shell`/`command`** (modules are idempotent; raw shell is not), and **always `--check --diff` before prod** (see what would change before you change it). Install Ansible, write a 10-line playbook that installs nginx, run it twice, and watch the second run report `changed=0` — once you've seen idempotency, the model clicks.