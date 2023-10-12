import torch
import torchvision
import matplotlib.pyplot as plt
import svgelements
import math

device = torch.device("cpu") # TODO: support mps, cuda if available

def load_obj(path):
    vertices = []
    faces = []
    with open(path) as file:
        for line in file:
            tokens = line.split(" ")
            if tokens[0] == "v":
                vertices.append([float(val) for val in tokens[1:]])
            elif tokens[0] == "f":
                faces.append([int(val)-1 for val in tokens[1:]])
    return (torch.tensor(vertices), faces)

def load_svg(path, sample_rate=8, out_width=2, out_height=2):
    lines = []
    svg = svgelements.SVG.parse(path)
    size = min(svg.width, svg.height)
    for element in svg.elements():
        if isinstance(element, svgelements.Path):
            pts = []
            length = element.length()
            if length == 0:
                continue
            num_lines = math.ceil(length / size * sample_rate)
            for i in range(num_lines + 1):
                pt = element.point(i / num_lines)
                pts.append([
                    pt.x / svg.width * out_width - out_width / 2,
                    -(pt.y / svg.height * out_height - out_height / 2)
                ])
            for i in range(num_lines):
                lines.append([pts[i], pts[i+1]])
    return torch.Tensor(lines)

def softclamp(tensor, min=0, max=1, hardness=10):
    return torch.special.expit((tensor - (max + min)/2) * hardness)

class Mesh:
    def __init__(self, models):
        self.scale = 0.5
        rest_vertices, faces = models[0]
        self.faces = faces
        self.rest_vertices = rest_vertices.to(device)
        self.vertices = rest_vertices.clone().to(device)
        # TODO make this a big tensor
        self.blend_shapes = [(v[0] - rest_vertices).clone().detach().to(device) for v in models[1:]]
        self.blend_weights = torch.tensor([0.0 for v in self.blend_shapes], requires_grad=True, device=device)
        self.triangles = torch.zeros([len(self.faces), 3, 3]).to(device)
        self.update_triangles()

    def constrain(self):
        torch.clamp(self.blend_weights, min=0, max=1, out=self.blend_weights)

    def update_triangles(self):
        self.vertices = self.rest_vertices.clone().detach().to(device)
        for i in range(self.blend_weights.size(dim=0)):
            self.vertices += self.blend_weights[i] * self.blend_shapes[i]
        self.triangles = torch.zeros([len(self.faces), 3, 3]).to(device)
        for i, face in enumerate(self.faces):
            for j in range(3):
                self.triangles[i, j, :] = self.vertices[face[j], :]

    def output(self, filename):
        obj = ""
        for i in range(self.vertices.size(dim=0)):
            obj += f"v {self.vertices[i,0]} {self.vertices[i,1]} {self.vertices[i,2]}\n"
        for face in self.faces:
            obj += f"f {int(face[0]+1)} {int(face[1]+1)} {int(face[2]+1)}\n"
        with open(filename, 'w') as file:
            file.write(obj)

    def cost(self, targets):
        pass

    def render(self, target=None, iter=0):
        alpha = self.brightness().cpu()

        target_costs = torch.zeros_like(alpha)
        target_weights = torch.zeros_like(alpha)
        target_alphas = torch.zeros_like(alpha)

        if target is not None:
            for i in range(target.size(dim=0)):
                alpha_cost, dist_weight, target_alpha = self.line_cost(alpha, i)
                target_costs += alpha_cost / target.size(dim=0)
                target_weights += dist_weight / target.size(dim=0)
                target_alphas += target_alpha / target.size(dim=0)

        plt.figure()
        for i in range(self.triangles.size(dim=0)):
            alpha_diff = target_alphas[i].item()-alpha[i].item()
            color = (max(0, alpha_diff), 0, max(0, -alpha_diff), alpha[i].item() * target_weights[i].item())
            triangle = plt.Polygon(self.triangles[i, :, 0:2].clone().detach().cpu(), color=color, ec='none')
            plt.gca().add_patch(triangle)
        plt.xlim(-2, 2)
        plt.ylim(-1, 1)

        if target is not None:
            target_cpu = target.clone().detach().cpu()
            for line in range(target_cpu.size(dim=0)):
                plt.plot(
                    (target_cpu[line,0,0].item(), target_cpu[line,1,0].item()),
                    (target_cpu[line,0,1].item(), target_cpu[line,1,1].item()),
                    'b'
                )
        plt.savefig(f"out/iter{iter}.png")
        plt.close()

    def brightness(self):
        to_camera = torch.Tensor([0, 0, 1]).to(device)
        t1 = self.triangles[:, 1, :] - self.triangles[:, 0, :]
        t2 = self.triangles[:, 2, :] - self.triangles[:, 0, :]
        normals = torch.linalg.cross(t1, t2)
        normals = normals / (torch.unsqueeze(torch.linalg.vector_norm(normals, dim=-1), dim=1).expand(normals.size()) + 1e-10)
        visibility = torch.linalg.vecdot(
            normals,
            to_camera.expand((normals.size(dim=0), 3))
        )
        brightness = torch.special.expit((visibility - 0.45) * 100)
        alpha = torch.special.expit((visibility) * 100) * (1 - brightness)
        return alpha

    def line_cost(self, alpha, i):
        v1 = target[i, 0, :]
        v2 = target[i, 1, :]
        line_dir = v2 - v1
        line_dir /= torch.linalg.norm(v2 - v1).expand(line_dir.size())
        line_len = torch.squeeze(torch.linalg.vecdot(v2 - v1, line_dir))
        line_normal = torch.stack([line_dir[1], -line_dir[0]], dim=-1)

        # line_dir and line_normal form an orthonormal basis aligned with the line.
        # We're going to transform all the triangle points into this space.

        to_points = self.triangles[:, :, 0:2] - v1.expand(self.triangles[:, :, 0:2].size())

        tangent_dists = torch.linalg.vecdot(to_points, line_dir.expand(to_points.size()))
        tangent_min = torch.min(tangent_dists, dim=1)[0]
        tangent_max = torch.max(tangent_dists, dim=1)[0]

        normal_dists = torch.linalg.vecdot(to_points, line_normal.expand(to_points.size()))
        min_normal = torch.min(normal_dists, dim=1)[0]
        max_normal = torch.max(normal_dists, dim=1)[0]

        # If a triangle is perfectly aligned, all vertices are a distance of 0 away in
        # the normal axis. Any deviation from this contributes to misalignment, so
        # we can add it to the total cost.
        min_normal_cost = torch.pow(min_normal, 2)
        max_normal_cost = torch.pow(max_normal, 2)
        normal_cost = min_normal_cost + max_normal_cost

        # The line itself has one endpoint at 0 and one at `line_len` along the tangent
        # axis. Any deviation from these contributes to triangles "sliding," so we can add
        # it to the total cost.
        min_tangent_costs = torch.pow(tangent_min, 2)
        max_tangent_costs = torch.pow(tangent_max - line_len.expand(tangent_max.size()), 2)
        tangent_cost = min_tangent_costs + max_tangent_costs

        # Get the minimum squared distance from the triangle to the line (x^2 + y^2, in
        # tangent-normal space)
        min_dist_sq = torch.min(torch.stack([torch.pow(tangent_min, 2), torch.pow(tangent_max - line_len.expand(tangent_max.size()), 2)]), dim=0)[0]
        min_dist_sq += torch.min(torch.stack([torch.pow(min_normal, 2), torch.pow(max_normal, 2)]), dim=0)[0]

        # We want the opacity (alpha) to be 1 when something it's on a line, and 0 if it's not. However,
        # we only care if it matches if it's actually close to the line. `alpha_dist_threshold` is the
        # minimum distance threshold we use.
        alpha_dist_threshold = 1
        # `target_alpha` should be 1 when the min distance is 0, and it should be 1 when the min
        # distance is >= `alpha_dist_threshold`
        target_alpha = -torch.special.expit((min_dist_sq - alpha_dist_threshold**2) * 100) + 1
        dist_cost = torch.special.expit((normal_cost + tangent_cost) * 1)
        alpha_cost = torch.pow(alpha - target_alpha, 2) * 0.1 + dist_cost * 10

        # We don't care about triangles that are very far away from the line, so we'll multiply
        # the weight for this triangle by this distance weight so that triangles whose min
        # distance is >= `dist_weight_threshold` don't have any cost added to it from this line.
        dist_weight_threshold = 2
        dist_weight = torch.special.expit((dist_weight_threshold**2 - min_dist_sq) * 1000)
        return (alpha_cost, dist_weight, target_alpha)

    def loss(self, target):
        loss = torch.tensor([0.0]).to(device)

        alpha = self.brightness()

        for i in range(target.size(dim=0)):
            alpha_cost, dist_weight, target_alpha = self.line_cost(alpha, i)
            loss += torch.sum(alpha_cost * dist_weight)

        # TODO regularize to prioritize zeroed blend weights

        return loss

suzanne = Mesh([
    load_obj("assets/suzanne0.obj"),
    load_obj("assets/suzanne1.obj"),
    load_obj("assets/suzanne2.obj")
])
target = load_svg("assets/guides.svg", out_width=4, out_height=2).to(device)

optimizer = torch.optim.Adam([suzanne.blend_weights], lr=2e-2) # 1e-2
# Run 100 Adam iterations.
for t in range(100):
    print('iteration:', t)
    optimizer.zero_grad()
    suzanne.update_triangles()
    loss = suzanne.loss(target)

    loss.backward()

    # Take a gradient descent step.
    optimizer.step()
    with torch.no_grad():
        # Force all weights to be between 0 and 1
        suzanne.constrain()
        suzanne.update_triangles()

    print(suzanne.blend_weights)
    suzanne.render(target, iter=t)

suzanne.output('blended.obj')
