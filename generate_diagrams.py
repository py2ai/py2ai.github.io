"""
Generate pydot diagrams for Claude Code blog posts
Uses orthogonal splines (90-degree angles) and SVG format
"""
import os
import pydot

OUTPUT_DIR = r"C:\Users\pc\Documents\trae_projects\smart\pyshine_web\py2ai.github.io\assets\img\posts\claude-code"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_complete_guide_diagram():
    graph = pydot.Dot(graph_type='digraph', rankdir='TB', bgcolor='white',
                      fontname='Helvetica', fontsize='14', splines='ortho',
                      nodesep='0.5', ranksep='0.75', dpi='150')
    
    graph.set_node_defaults(shape='box', style='filled,rounded', 
                            fillcolor='#E3F2FD', fontname='Helvetica', fontsize='12',
                            width='2', height='0.6')
    graph.set_edge_defaults(color='#1976D2', fontname='Helvetica', fontsize='10',
                            arrowsize='0.8')
    
    user = pydot.Node('user', label='User Input\n(Prompt/Command)', shape='ellipse', 
                      fillcolor='#FFF3E0', style='filled', width='1.8')
    graph.add_node(user)
    
    claude = pydot.Node('claude', label='Claude Code\n(CLI)', shape='box3d',
                        fillcolor='#4CAF50', fontcolor='white', style='filled', width='1.5')
    graph.add_node(claude)
    
    graph.add_edge(pydot.Edge(user, claude, label=' sends'))
    
    tools = pydot.Node('tools', label='Tool System\n(Bash, Read, Write,\nEdit, Grep, etc.)',
                      shape='box', fillcolor='#E8F5E9', width='2.2')
    files = pydot.Node('files', label='File System\n(Local Files)',
                      shape='folder', fillcolor='#FFF8E1', width='1.8')
    mcp = pydot.Node('mcp', label='MCP Servers\n(External APIs)',
                    shape='component', fillcolor='#F3E5F5', width='1.8')
    graph.add_node(tools)
    graph.add_node(files)
    graph.add_node(mcp)
    
    graph.add_edge(pydot.Edge(claude, tools, label=' uses'))
    graph.add_edge(pydot.Edge(tools, files, label=' reads/writes'))
    graph.add_edge(pydot.Edge(claude, mcp, label=' connects'))
    
    memory = pydot.Node('memory', label='CLAUDE.md\n(Memory)', shape='note',
                        fillcolor='#FFECB3', width='1.5')
    graph.add_node(memory)
    graph.add_edge(pydot.Edge(memory, claude, label=' provides context', style='dashed'))
    
    output = pydot.Node('output', label='Output\n(Code, Files,\nExplanations)',
                        shape='ellipse', fillcolor='#E8F5E9', width='1.8')
    graph.add_node(output)
    graph.add_edge(pydot.Edge(claude, output, label=' produces'))
    
    output_path = os.path.join(OUTPUT_DIR, "claude-code-architecture.svg")
    graph.write_svg(output_path)
    print(f"Created: {output_path}")
    return output_path


def create_slash_commands_diagram():
    graph = pydot.Dot(graph_type='digraph', rankdir='LR', bgcolor='white',
                      fontname='Helvetica', fontsize='14', splines='ortho',
                      nodesep='0.6', ranksep='0.8', dpi='150')
    
    graph.set_node_defaults(shape='box', style='filled,rounded',
                            fillcolor='#E3F2FD', fontname='Helvetica', fontsize='12',
                            width='2', height='0.6')
    graph.set_edge_defaults(color='#1976D2', fontname='Helvetica', fontsize='10',
                            arrowsize='0.8')
    
    user = pydot.Node('user', label='User types\n/command', shape='ellipse',
                      fillcolor='#FFF3E0', width='1.5')
    graph.add_node(user)
    
    parser = pydot.Node('parser', label='Command Parser', shape='box',
                        fillcolor='#BBDEFB', width='1.8')
    graph.add_node(parser)
    graph.add_edge(pydot.Edge(user, parser))
    
    built_in = pydot.Node('built_in', label='Built-in Commands\n/help, /clear,\n/model, /config',
                         shape='box', fillcolor='#C8E6C9', width='2.2')
    custom = pydot.Node('custom', label='Custom Commands\n.claude/commands/\n*.md files',
                       shape='box', fillcolor='#FFCCBC', width='2.2')
    mcp_cmd = pydot.Node('mcp_cmd', label='MCP Commands\n/mcp__server__tool',
                        shape='box', fillcolor='#D1C4E9', width='2.2')
    graph.add_node(built_in)
    graph.add_node(custom)
    graph.add_node(mcp_cmd)
    
    graph.add_edge(pydot.Edge(parser, built_in, label=' built-in'))
    graph.add_edge(pydot.Edge(parser, custom, label=' custom'))
    graph.add_edge(pydot.Edge(parser, mcp_cmd, label=' mcp'))
    
    action = pydot.Node('action', label='Execute Action', shape='box3d',
                        fillcolor='#4CAF50', fontcolor='white', width='1.8')
    graph.add_node(action)
    graph.add_edge(pydot.Edge(built_in, action))
    graph.add_edge(pydot.Edge(custom, action))
    graph.add_edge(pydot.Edge(mcp_cmd, action))
    
    output_path = os.path.join(OUTPUT_DIR, "slash-commands-flow.svg")
    graph.write_svg(output_path)
    print(f"Created: {output_path}")
    return output_path


def create_memory_diagram():
    graph = pydot.Dot(graph_type='digraph', rankdir='TB', bgcolor='white',
                      fontname='Helvetica', fontsize='14', splines='ortho',
                      nodesep='0.5', ranksep='0.75', dpi='150')
    
    graph.set_node_defaults(shape='box', style='filled,rounded',
                            fillcolor='#E3F2FD', fontname='Helvetica', fontsize='12',
                            width='2', height='0.6')
    graph.set_edge_defaults(color='#1976D2', fontname='Helvetica', fontsize='10',
                            arrowsize='0.8')
    
    memory_file = pydot.Node('memory_file', label='CLAUDE.md\n(Memory File)',
                             shape='note', fillcolor='#FFECB3', style='filled', width='1.8')
    graph.add_node(memory_file)
    
    project = pydot.Node('project', label='Project Memory\n./CLAUDE.md',
                        shape='note', fillcolor='#C8E6C9', width='2')
    user = pydot.Node('user', label='User Memory\n~/.claude/CLAUDE.md',
                     shape='note', fillcolor='#BBDEFB', width='2.2')
    graph.add_node(project)
    graph.add_node(user)
    
    graph.add_edge(pydot.Edge(memory_file, project))
    graph.add_edge(pydot.Edge(memory_file, user))
    
    claude = pydot.Node('claude', label='Claude Code\nSession', shape='box3d',
                        fillcolor='#4CAF50', fontcolor='white', width='1.8')
    graph.add_node(claude)
    graph.add_edge(pydot.Edge(project, claude, label=' loads'))
    graph.add_edge(pydot.Edge(user, claude, label=' loads'))
    
    context = pydot.Node('context', label='Context Window\n(Project Info,\nPreferences,\nConventions)',
                         shape='box', fillcolor='#F3E5F5', width='2.2')
    graph.add_node(context)
    graph.add_edge(pydot.Edge(claude, context, label=' builds'))
    
    output = pydot.Node('output', label='Context-Aware\nResponses', shape='ellipse',
                        fillcolor='#E8F5E9', width='1.8')
    graph.add_node(output)
    graph.add_edge(pydot.Edge(context, output, label=' enables'))
    
    output_path = os.path.join(OUTPUT_DIR, "memory-system-flow.svg")
    graph.write_svg(output_path)
    print(f"Created: {output_path}")
    return output_path


def create_skills_diagram():
    graph = pydot.Dot(graph_type='digraph', rankdir='TB', bgcolor='white',
                      fontname='Helvetica', fontsize='14', splines='ortho',
                      nodesep='0.5', ranksep='0.75', dpi='150')
    
    graph.set_node_defaults(shape='box', style='filled,rounded',
                            fillcolor='#E3F2FD', fontname='Helvetica', fontsize='12',
                            width='2', height='0.6')
    graph.set_edge_defaults(color='#1976D2', fontname='Helvetica', fontsize='10',
                            arrowsize='0.8')
    
    skill_file = pydot.Node('skill_file', label='Skill Definition\n.md file',
                            shape='note', fillcolor='#FFECB3', width='1.8')
    graph.add_node(skill_file)
    
    trigger = pydot.Node('trigger', label='Trigger\n(DESCRIPTION in\nfrontmatter)',
                        shape='box', fillcolor='#BBDEFB', width='2.2')
    prompt = pydot.Node('prompt', label='Prompt Template\n(Markdown body)',
                       shape='box', fillcolor='#C8E6C9', width='2')
    graph.add_node(trigger)
    graph.add_node(prompt)
    
    graph.add_edge(pydot.Edge(skill_file, trigger))
    graph.add_edge(pydot.Edge(skill_file, prompt))
    
    user = pydot.Node('user', label='User Request\nmatches trigger',
                      shape='ellipse', fillcolor='#FFF3E0', width='2')
    graph.add_node(user)
    
    matcher = pydot.Node('matcher', label='Skill Matcher', shape='diamond',
                         fillcolor='#FFCCBC', width='1.5')
    graph.add_node(matcher)
    graph.add_edge(pydot.Edge(user, matcher))
    graph.add_edge(pydot.Edge(trigger, matcher, style='dashed'))
    
    claude = pydot.Node('claude', label='Claude Code\nExecutes Skill',
                        shape='box3d', fillcolor='#4CAF50', fontcolor='white', width='2')
    graph.add_node(claude)
    graph.add_edge(pydot.Edge(matcher, claude, label=' invokes'))
    graph.add_edge(pydot.Edge(prompt, claude, label=' provides instructions'))
    
    output = pydot.Node('output', label='Skill Output\n(Completed Task)',
                        shape='ellipse', fillcolor='#E8F5E9', width='2')
    graph.add_node(output)
    graph.add_edge(pydot.Edge(claude, output))
    
    output_path = os.path.join(OUTPUT_DIR, "skills-workflow.svg")
    graph.write_svg(output_path)
    print(f"Created: {output_path}")
    return output_path


def create_subagents_diagram():
    graph = pydot.Dot(graph_type='digraph', rankdir='TB', bgcolor='white',
                      fontname='Helvetica', fontsize='14', splines='ortho',
                      nodesep='0.5', ranksep='0.75', dpi='150')
    
    graph.set_node_defaults(shape='box', style='filled,rounded',
                            fillcolor='#E3F2FD', fontname='Helvetica', fontsize='12',
                            width='2', height='0.6')
    graph.set_edge_defaults(color='#1976D2', fontname='Helvetica', fontsize='10',
                            arrowsize='0.8')
    
    main = pydot.Node('main', label='Main Agent\n(Claude Code)',
                      shape='box3d', fillcolor='#4CAF50', fontcolor='white', width='2')
    graph.add_node(main)
    
    architect = pydot.Node('architect', label='Architect Agent\n(System Design)',
                          shape='component', fillcolor='#BBDEFB', width='2')
    engineer = pydot.Node('engineer', label='Engineer Agent\n(Code Implementation)',
                         shape='component', fillcolor='#C8E6C9', width='2.2')
    reviewer = pydot.Node('reviewer', label='Reviewer Agent\n(Code Review)',
                         shape='component', fillcolor='#FFCCBC', width='2')
    tester = pydot.Node('tester', label='Tester Agent\n(Testing)',
                       shape='component', fillcolor='#D1C4E9', width='1.8')
    graph.add_node(architect)
    graph.add_node(engineer)
    graph.add_node(reviewer)
    graph.add_node(tester)
    
    graph.add_edge(pydot.Edge(main, architect, label=' delegates'))
    graph.add_edge(pydot.Edge(main, engineer, label=' delegates'))
    graph.add_edge(pydot.Edge(main, reviewer, label=' delegates'))
    graph.add_edge(pydot.Edge(main, tester, label=' delegates'))
    
    results = pydot.Node('results', label='Aggregated Results\n(Completed Project)',
                         shape='box', fillcolor='#F3E5F5', width='2.2')
    graph.add_node(results)
    graph.add_edge(pydot.Edge(architect, results))
    graph.add_edge(pydot.Edge(engineer, results))
    graph.add_edge(pydot.Edge(reviewer, results))
    graph.add_edge(pydot.Edge(tester, results))
    
    output_path = os.path.join(OUTPUT_DIR, "subagents-architecture.svg")
    graph.write_svg(output_path)
    print(f"Created: {output_path}")
    return output_path


def create_mcp_diagram():
    graph = pydot.Dot(graph_type='digraph', rankdir='LR', bgcolor='white',
                      fontname='Helvetica', fontsize='14', splines='ortho',
                      nodesep='0.4', ranksep='0.6', dpi='150')
    
    graph.set_node_defaults(shape='box', style='filled,rounded',
                            fillcolor='#E3F2FD', fontname='Helvetica', fontsize='12',
                            width='1.8', height='0.6')
    graph.set_edge_defaults(color='#1976D2', fontname='Helvetica', fontsize='10',
                            arrowsize='0.8')
    
    claude = pydot.Node('claude', label='Claude Code', shape='box3d',
                        fillcolor='#4CAF50', fontcolor='white', width='1.5')
    graph.add_node(claude)
    
    mcp_layer = pydot.Node('mcp_layer', label='MCP Protocol Layer',
                           shape='box', fillcolor='#BBDEFB', style='filled,dashed', width='2')
    graph.add_node(mcp_layer)
    graph.add_edge(pydot.Edge(claude, mcp_layer, label=' communicates'))
    
    github = pydot.Node('github', label='GitHub MCP\n(PR, Issues,\nRepos)',
                       shape='component', fillcolor='#C8E6C9', width='1.8')
    db = pydot.Node('db', label='Database MCP\n(PostgreSQL,\nMySQL)',
                   shape='component', fillcolor='#FFCCBC', width='1.8')
    slack = pydot.Node('slack', label='Slack MCP\n(Messages,\nChannels)',
                      shape='component', fillcolor='#D1C4E9', width='1.8')
    fs = pydot.Node('fs', label='Filesystem MCP\n(Files,\nDirectories)',
                   shape='component', fillcolor='#FFECB3', width='1.8')
    graph.add_node(github)
    graph.add_node(db)
    graph.add_node(slack)
    graph.add_node(fs)
    
    graph.add_edge(pydot.Edge(mcp_layer, github))
    graph.add_edge(pydot.Edge(mcp_layer, db))
    graph.add_edge(pydot.Edge(mcp_layer, slack))
    graph.add_edge(pydot.Edge(mcp_layer, fs))
    
    gh_api = pydot.Node('gh_api', label='GitHub API', shape='ellipse',
                       fillcolor='#E8F5E9', width='1.3')
    db_api = pydot.Node('db_api', label='Database', shape='cylinder',
                       fillcolor='#E8F5E9', width='1.2')
    slack_api = pydot.Node('slack_api', label='Slack API', shape='ellipse',
                          fillcolor='#E8F5E9', width='1.2')
    fs_api = pydot.Node('fs_api', label='Local Files', shape='folder',
                       fillcolor='#E8F5E9', width='1.2')
    graph.add_node(gh_api)
    graph.add_node(db_api)
    graph.add_node(slack_api)
    graph.add_node(fs_api)
    
    graph.add_edge(pydot.Edge(github, gh_api))
    graph.add_edge(pydot.Edge(db, db_api))
    graph.add_edge(pydot.Edge(slack, slack_api))
    graph.add_edge(pydot.Edge(fs, fs_api))
    
    output_path = os.path.join(OUTPUT_DIR, "mcp-architecture.svg")
    graph.write_svg(output_path)
    print(f"Created: {output_path}")
    return output_path


def create_hooks_diagram():
    graph = pydot.Dot(graph_type='digraph', rankdir='TB', bgcolor='white',
                      fontname='Helvetica', fontsize='14', splines='ortho',
                      nodesep='0.4', ranksep='0.6', dpi='150')
    
    graph.set_node_defaults(shape='box', style='filled,rounded',
                            fillcolor='#E3F2FD', fontname='Helvetica', fontsize='12',
                            width='2', height='0.6')
    graph.set_edge_defaults(color='#1976D2', fontname='Helvetica', fontsize='10',
                            arrowsize='0.8')
    
    event = pydot.Node('event', label='Hook Event\n(PreToolUse, PostToolUse,\nNotification, etc.)',
                       shape='ellipse', fillcolor='#FFF3E0', width='2.5')
    graph.add_node(event)
    
    matcher = pydot.Node('matcher', label='Pattern Matcher\n(Tool Name,\nFile Pattern)',
                         shape='diamond', fillcolor='#FFCCBC', width='2')
    graph.add_node(matcher)
    graph.add_edge(pydot.Edge(event, matcher, label=' triggers'))
    
    cmd_hook = pydot.Node('cmd_hook', label='Command Hook\n(Shell Script)',
                         shape='box', fillcolor='#C8E6C9', width='2')
    http_hook = pydot.Node('http_hook', label='HTTP Hook\n(Webhook)',
                          shape='box', fillcolor='#BBDEFB', width='1.8')
    prompt_hook = pydot.Node('prompt_hook', label='Prompt Hook\n(Inject Context)',
                            shape='box', fillcolor='#D1C4E9', width='2')
    agent_hook = pydot.Node('agent_hook', label='Agent Hook\n(Delegate Task)',
                           shape='box', fillcolor='#FFECB3', width='2')
    graph.add_node(cmd_hook)
    graph.add_node(http_hook)
    graph.add_node(prompt_hook)
    graph.add_node(agent_hook)
    
    graph.add_edge(pydot.Edge(matcher, cmd_hook))
    graph.add_edge(pydot.Edge(matcher, http_hook))
    graph.add_edge(pydot.Edge(matcher, prompt_hook))
    graph.add_edge(pydot.Edge(matcher, agent_hook))
    
    decision = pydot.Node('decision', label='Decision\n(allow/block/modify)',
                          shape='diamond', fillcolor='#F3E5F5', width='2')
    graph.add_node(decision)
    graph.add_edge(pydot.Edge(cmd_hook, decision))
    graph.add_edge(pydot.Edge(http_hook, decision))
    graph.add_edge(pydot.Edge(prompt_hook, decision))
    graph.add_edge(pydot.Edge(agent_hook, decision))
    
    allow = pydot.Node('allow', label='Allow\n(continue)', shape='box',
                      fillcolor='#C8E6C9', width='1.5')
    block = pydot.Node('block', label='Block\n(stop)', shape='box',
                      fillcolor='#FFCDD2', width='1.5')
    modify = pydot.Node('modify', label='Modify\n(change input)', shape='box',
                       fillcolor='#BBDEFB', width='1.8')
    graph.add_node(allow)
    graph.add_node(block)
    graph.add_node(modify)
    
    graph.add_edge(pydot.Edge(decision, allow))
    graph.add_edge(pydot.Edge(decision, block))
    graph.add_edge(pydot.Edge(decision, modify))
    
    output_path = os.path.join(OUTPUT_DIR, "hooks-flow.svg")
    graph.write_svg(output_path)
    print(f"Created: {output_path}")
    return output_path


if __name__ == "__main__":
    print("Generating Claude Code diagrams (SVG with orthogonal splines)...")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    create_complete_guide_diagram()
    create_slash_commands_diagram()
    create_memory_diagram()
    create_skills_diagram()
    create_subagents_diagram()
    create_mcp_diagram()
    create_hooks_diagram()
    
    print()
    print("All diagrams generated successfully!")
