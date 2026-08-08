---
layout: post
title: "Learn Packer in a Single Post: A Complete Tutorial From Builders and Provisioners to Immutable Images and CI/CD"
description: "A complete HashiCorp Packer tutorial in one blog post. Covers the whole tool in 5 stages: templates (HCL2, source/build blocks, variables), builders (amazon-ebs, qemu, virtualbox, docker, azure, gcp), provisioners (shell, ansible, file, windows-restart, breakpoint debugging), post-processors (compress, vagrant, import), and production (CI/CD integration, source images, immutable infrastructure, blue/green, vs Docker comparison). Five hand-drawn diagrams, runnable HCL, and a quick-start roadmap."
date: 2026-07-28
header-img: "img/post-bg.jpg"
permalink: /Learn-Packer-in-One-Post-Complete-Tutorial-Builders-Provisioners-Immutable-Images-Quick-Start/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - Packer
  - Immutable Infrastructure
  - DevOps
  - IaC
  - AMI
  - Tutorial
categories: [Tutorial, DevOps, Infrastructure]
keywords: "HashiCorp Packer tutorial one post, learn Packer fast, Packer template HCL2 source build blocks variables, Packer builders amazon-ebs qemu virtualbox docker azure gcp, Packer provisioners shell ansible file windows-restart, Packer post-processors compress vagrant import, Packer CI/CD GitHub Actions, Packer immutable infrastructure blue green, Packer vs Docker, Packer source images golden image, Packer quick start roadmap"
author: "PyShine"
image: /assets/img/diagrams/ansible-tutorial/ans-flow.svg
---

# Learn Packer in a Single Post: Complete Tutorial From Builders and Provisioners to Immutable Images

Packer is HashiCorp's tool for building **machine images** — AMIs for AWS, VHDs for Azure, GCE images for GCP, Vagrant boxes, Docker images, VMware VMs — all from a single declarative template. Its value: build a **golden image** once, deploy N identical copies, and never patch a running server again (update = build a new image + redeploy). This is **immutable infrastructure**, and it eliminates configuration drift. This single post covers the whole tool in five stages, with hand-drawn diagrams and runnable HCL.

## Learning Roadmap

![Packer Roadmap](/assets/img/diagrams/packer-tutorial/pkr-roadmap.svg)

The roadmap moves from templates (Stage 1), through builders (Stage 2), provisioners (Stage 3), post-processors (Stage 4), and production (Stage 5). The [Terraform tutorial](/Learn-Terraform-in-One-Post-Complete-Tutorial-HCL-State-Modules-Providers-Quick-Start/) and [Ansible tutorial](/Learn-Ansible-in-One-Post-Complete-Tutorial-Inventory-Playbooks-Roles-Vault-Quick-Start/) are companions — Packer builds the images Terraform deploys and Ansible could provision.

---

## Stage 1 — Templates (HCL2)

### The source + build model

![Packer Build Pipeline: Immutable Artifacts](/assets/img/diagrams/packer-tutorial/pkr-pipeline.svg)

Packer templates (since v1.7, written in **HCL2** — the same language as Terraform) define a **source** (what to start from) and a **build** (how to customize it):

```hcl
# ami.pkr.hcl
packer {
  required_plugins {
    amazon = { version = ">= 1.2.0", source = "github.com/hashicorp/amazon" }
  }
}

variable "region" { default = "us-east-1" }
variable "version" { default = "1.0.0" }

source "amazon-ebs" "ubuntu" {
  region          = var.region
  source_ami      = "ami-0c7217cdde317cfec"   # Ubuntu 22.04 base
  instance_type   = "t3.small"
  ssh_username    = "ubuntu"
  ami_name        = "my-app-${var.version}-{{timestamp}}"
}

build {
  sources = ["source.amazon-ebs.ubuntu"]

  provisioner "shell" {
    script = "install-app.sh"
  }
}
```

- **`source` block** — the builder type (`amazon-ebs`) + a name (`ubuntu`) + the base image + instance config.
- **`build` block** — references the source(s) + the provisioners that customize it.
- **`variables`** — parameterize the template (region, version).
- **`{{timestamp}}`** — a template engine function, makes each build a unique AMI name.

```bash
packer init .       # install plugins
packer validate .   # check the template
packer build .      # build the image
```

> **Pitfall:** Old Packer used JSON templates; new Packer (1.7+) uses HCL2. The JSON format still works but is deprecated. Write new templates in HCL2 — it's the same language as Terraform, supports variables/conditionals/loops, and is far more readable.

---

## Stage 2 — Builders

### Where the image is created

![Builders: Where the Image Is Created](/assets/img/diagrams/packer-tutorial/pkr-builders.svg)

A **builder** creates the image on a specific platform. The builder launches a temporary instance, runs the provisioners, snapshots it, and saves the image:

| Builder | Platform | Use |
|---|---|---|
| **`amazon-ebs`** | AWS (launch EC2, snapshot to AMI) | the common AWS builder |
| **`amazon-instance`** | AWS (build on a temp instance) | rare, complex setups |
| **`azure-arm`** | Azure (managed image) | Azure |
| **`googlecompute`** | GCP (GCE image) | GCP |
| **`qemu` / `virtualbox`** | local VM (KVM/VirtualBox) | dev, Vagrant boxes |
| **`docker`** | Docker image (from Dockerfile) | the docker builder |
| **`vsphere` / `hyperv`** | VMware/Hyper-V | on-prem virtualization |
| **`null`** | no builder — run provisioners on an existing host | debugging |

The `amazon-ebs` builder: launches an EC2 instance from the `source_ami`, runs the provisioners on it, stops it, creates an EBS snapshot, and registers it as a new AMI. The temporary instance is then terminated — you're left with just the AMI.

---

## Stage 3 — Provisioners

### Customize the image

![Provisioners: Customize the Image](/assets/img/diagrams/packer-tutorial/pkr-provisioners.svg)

A **provisioner** runs inside the building instance to install software, copy files, and configure the OS:

```hcl
build {
  sources = ["source.amazon-ebs.ubuntu"]

  provisioner "shell" {
    inline = [
      "sudo apt-get update",
      "sudo apt-get install -y nginx",
      "sudo systemctl enable nginx",
    ]
  }

  provisioner "file" {
    source      = "app.conf"
    destination = "/tmp/app.conf"
  }

  provisioner "shell" {
    script = "configure-app.sh"
  }
}
```

### Common provisioners

| Provisioner | What it does |
|---|---|
| **`shell`** | run shell scripts/commands in the image (install packages, configure) |
| **`shell` (local)** | run shell on the *host* running Packer (not in the image) |
| **`ansible`** | run an Ansible playbook against the building instance |
| **`ansible-local`** | run Ansible *inside* the image (no control node needed) |
| **`file`** | copy files into the image |
| **`puppet`** | run a Puppet manifest |
| **`windows-restart`** | restart Windows (Windows Updates need a reboot) |
| **`breakpoint`** | pause for manual debugging (SSH into the build) |

### Reusing Ansible

```hcl
provisioner "ansible" {
  playbook_file = "./site.yml"
  ansible_env_vars = ["ANSIBLE_HOST_KEY_CHECKING=False"]
}
```

The `ansible` provisioner runs an existing Ansible playbook against the building instance — so your Packer image and your Ansible-managed servers share the same configuration code. This is the bridge between the "golden image" and "config management" approaches: build the image with Packer, provision it with Ansible, deploy it with Terraform.

> **Pitfall:** Provisioners run in order, and a failure stops the build. If `apt-get install` fails, the whole build fails — which is what you want (don't ship a broken image). But a flaky network during `apt-get update` can fail a build; add retries (`max_retries`) for network-dependent provisioners.

---

## Stage 4 — Post-Processors

### Transform the artifact after the build

A **post-processor** runs *after* the build, transforming the artifact:

```hcl
build {
  sources = ["source.amazon-ebs.ubuntu"]
  provisioner "shell" { script = "install.sh" }

  post-processor "vagrant" {
    keep_input_artifact = true   # keep the AMI AND create a Vagrant box
  }

  post-processor "compress" {
    output = "my-app.tar.gz"
  }
}
```

| Post-processor | What it does |
|---|---|
| **`compress`** | compress the artifact (tar.gz) |
| **`vagrant`** | create a Vagrant box from the build |
| **`amazon-import`** | import a local image into AWS as an AMI |
| **`docker-import`** | import a tarball as a Docker image |
| **`docker-push`** | push a Docker image to a registry |
| **`artifice`** | keep a specific artifact file |
| **`shell-local`** | run a shell command on the host after build |

Post-processors can be **chained** — build an AMI → create a Vagrant box from it → compress the box → push to a registry. One template, multiple artifact formats.

---

## Stage 5 — Production

![Production: CI/CD, Source Images, Immutable](/assets/img/diagrams/packer-tutorial/pkr-prod.svg)

### CI/CD integration

```yaml
# .github/workflows/build-ami.yml
name: Build AMI
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: hashicorp/setup-packer@v3
      - run: packer init .
      - run: packer validate .
      - run: packer build .
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_KEY }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET }}
```

Run `packer build` in CI (GitHub Actions, Jenkins, GitLab CI) on every push or on a schedule. Always run `packer validate` first (catches template errors before launching a paid EC2 instance).

### Source images — the golden base

```hcl
source "amazon-ebs" "ubuntu" {
  source_ami = "ami-0c7217cdde317cfec"   # the official Ubuntu 22.04 AMI
  # OR a custom internal base AMI:
  # source_ami = "ami-12345678"            # your org's hardened base
}
```

The `source_ami` is your starting point. Common patterns:
- **Official base** (Ubuntu, Amazon Linux, Windows) — simple, but you install everything each build.
- **Custom internal base** — a pre-hardened AMI (CIS benchmarks, security agents, org defaults) that *your* Packer build extends. Build the base once, extend it for each app. This is the "golden image" pattern — two layers (org base → app image).

### Immutable infrastructure

The philosophy: **never modify a running server**. To update:
1. Build a **new** AMI with Packer (new app version, new security patches).
2. Deploy new instances from the new AMI (via Terraform / ASG).
3. **Blue/green** — bring up the new instances, verify, then swap traffic and terminate the old ones.
4. **Rollback** = launch the old AMI again (instant revert).

This eliminates **configuration drift** — every server is a bit-for-bit copy of the image. No "this one server was SSH'd into and manually patched" surprises. The same image runs in dev, staging, and prod (the only difference is config injected at boot via user-data/env vars).

### Multi-arch + manifests

```hcl
# build amd64 and arm64, combine into one manifest
build {
  sources = [
    "source.amazon-ebs.ubuntu-amd64",
    "source.amazon-ebs.ubuntu-arm64",
  ]
  # ... provisioners ...
  post-processor "manifest" {
    output = "manifest.json"
  }
}
```

Build for multiple architectures (amd64 + arm64 — important for AWS Graviton), and combine them into a single manifest so the deployment picks the right architecture per instance type.

### Packer vs Docker

| | Packer | Docker |
|---|---|---|
| **What it builds** | full OS image (AMI, VM) — includes the kernel | app + deps in a container — shares the host kernel |
| **Boots independently** | ✅ (full VM/AMI) | ❌ (needs a host OS + container runtime) |
| **Isolation** | full VM isolation | process-level isolation (shared kernel) |
| **Use case** | EC2 instances, VMs, base images | app packaging, microservices |
| **Relationship** | can build the Docker host image | Docker can be a Packer builder |

They're complementary: **Packer builds the VM/AMI; Docker runs the containers on it.** Or Packer's `docker` builder produces a Docker image directly. They solve different problems — Packer for "what does the whole machine look like," Docker for "how is this app packaged."

> **Pitfall:** Packer builds are **slow and cost money** — each build launches a real EC2 instance (billed per second), runs provisioners, and snapshots. Don't build on every commit; build on a schedule (nightly) or on version bumps. Cache the early provisioner steps (install base packages) in a custom source AMI so the app-image build only does the app-specific steps.

---

## Quick-Start Checklist

1. **Install Packer** — `brew install packer` / download from hashicorp.com.
2. **Write a template** — `source "amazon-ebs"` from an Ubuntu AMI + one `shell` provisioner.
3. **`packer init .`** — install the required plugins.
4. **`packer validate .`** — check the template (always do this first).
5. **`packer build .`** — build the AMI (watch the EC2 instance launch in the AWS console).
6. **Launch an EC2 instance** from the new AMI — verify nginx (or whatever) is installed.
7. **Parameterize** with `variables` (region, version, app_version).
8. **Add an Ansible provisioner** to reuse your existing playbooks.
9. **Run in CI** — GitHub Actions `packer build` on push.
10. **Adopt immutable infra** — build new AMIs, deploy via ASG, never patch in place.

## Common Pitfalls

- **JSON templates (deprecated)** — write HCL2, not JSON; HCL2 has variables/loops/conditionals.
- **No `packer validate` before build** — a template error launches a paid instance that fails; validate first.
- **Building on every commit** — builds are slow + cost money; build on schedule or version bump.
- **Forgetting `{{timestamp}}` in AMI name** — without it, every build tries to create an AMI with the same name → conflict.
- **Patching running servers** — defeats the purpose; update = new image + redeploy, never SSH + patch.
- **No custom source AMI** — installing base packages every build is slow; build a hardened base once, extend it.
- **Flaky provisioners without retries** — `apt-get update` can fail on network; add `max_retries`.
- **Not cleaning up old AMIs** — old AMIs + snapshots accumulate AWS costs; add a cleanup script.

## Further Reading

- [Packer Docs](https://developer.hashicorp.com/packer/docs) — the official reference
- [Packer Templates (HCL2)](https://developer.hashicorp.com/packer/templates) — the template language
- [Packer Builders](https://developer.hashicorp.com/packer/plugins/builders) — all builder types
- [Immutable Infrastructure](https://www.hashicorp.com/resources/immutable-infrastructure-in-practice) — the philosophy
- [Packer + Ansible](https://developer.hashicorp.com/packer/plugins/provisioners/ansible/ansible) — the bridge

## Related guides

Packer is the image-building layer of the DevOps stack — these PyShine tutorials connect to it:

- **[Learn Terraform in One Post](/Learn-Terraform-in-One-Post-Complete-Tutorial-HCL-State-Modules-Providers-Quick-Start/)** — Terraform deploys the images Packer builds; both use HCL2.
- **[Learn Ansible in One Post](/Learn-Ansible-in-One-Post-Complete-Tutorial-Inventory-Playbooks-Roles-Vault-Quick-Start/)** — the `ansible` provisioner runs your playbooks inside the Packer build.
- **[Learn Docker in One Post](/Learn-Docker-in-One-Post-Complete-Tutorial-Dockerfile-Volumes-Compose-Quick-Start/)** — Packer builds VMs/AMIs (full OS); Docker builds container images (app layer). Complementary.
- **[Learn Kubernetes in One Post](/Learn-Kubernetes-in-One-Post-Complete-Tutorial-Pods-Deployments-Services-Production-Quick-Start/)** — Packer builds the K8s node AMIs; Docker builds the pod images.
- **[Learn GitHub Actions in One Post](/Learn-GitHub-Actions-in-One-Post-Complete-Tutorial-Workflows-Jobs-Runners-Matrix-Quick-Start/)** — run `packer build` in CI on every push.

---

Packer's value is **building identical, reproducible machine images** — one template produces AMIs, Vagrant boxes, Docker images, and VMs, and the resulting images enable immutable infrastructure (never patch, always rebuild + redeploy). The five stages here — templates, builders, provisioners, post-processors, production — cover everything from a single `shell` provisioner to a multi-arch, CI-driven, Ansible-provisioned, golden-base-extended production image pipeline. The two habits that pay off: **always `packer validate` before `packer build`** (a template error shouldn't cost you a paid EC2 instance), and **adopt the golden-base pattern** (build a hardened org base AMI once, extend it per app — don't install base packages every build). Write a 20-line HCL2 template, `packer build` an AMI, launch an EC2 instance from it, and SSH in to find nginx already installed — once you've seen the image boot pre-configured, the immutable-infra model clicks.