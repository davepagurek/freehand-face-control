import torch
import torchvision
import matplotlib.pyplot as plt
import svgelements
import math

device = torch.device("mps") # TODO: change to "mps" for macbook

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

# Calculates the signed distance from the edge v0-v1 to point p
# https://gist.github.com/seece/7e2cc8f37b2446035c27cbc43a561ddc
def signed_dist_to_edge(v0, v1, p):
    """
    let S = W*H
    v0 and v1 have vertex positions for all N triangles.
    Their shapes are [N x 2]
    p is a list of sampling points as a [N x S x 2] tensor.
    Each of the N triangles has an [S x 2] matrix of sampling points.
    returns a [N x S] matrix
    """

    S = p.size()[1]

    # Take all the x and y coordinates of all the positions as a
    # [N x S] tensor
    px = p[:, :, 0].cuda()
    py = p[:, :, 1].cuda()

    # We need to manually broadcast the vector to cover all sample points
    y01 = v0[:,1] - v1[:,1] # [N]
    x10 = v1[:,0] - v0[:,0] # [N]
    y01 = y01.unsqueeze(0).t().repeat((1, S)).cuda() # [N x S]
    x10 = x10.unsqueeze(0).t().repeat((1, S)).cuda() # [N x S]

    cross = v0[:,0]*v1[:,1] - v0[:,1]*v1[:,0] # [N]
    cross = cross.unsqueeze(0).t().repeat((1,S)) # [N x S]

    return y01*px + x10*py + cross

class Mesh:
    def __init__(self, models):
        self.scale = 0.5
        rest_vertices, faces = models[0]
        self.faces = faces
        self.rest_vertices = rest_vertices.to(device)
        self.vertices = rest_vertices.clone().to(device)
        # TODO make this a big tensor
        self.blend_shapes = [(v[0] - rest_vertices).detach().to(device) for v in models[1:]]
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

    def display(self, target=None):
        plt.figure()
        to_camera = torch.Tensor([0, 0, 1]).cpu()
        for i in range(self.triangles.size(dim=0)):
            v1 = self.triangles[i, 0, :].clone().detach().cpu()
            v2 = self.triangles[i, 1, :].clone().detach().cpu()
            v3 = self.triangles[i, 2, :].clone().detach().cpu()
            t1 = v2 - v1
            t2 = v3 - v1
            n = torch.linalg.cross(t1, t2)
            n = n / (torch.linalg.vector_norm(n) + 1e-10)
            visibility = torch.dot(n, to_camera)
            brightness = torch.clamp(visibility, min=0, max=1).pow(0.25).item()
            alpha = (torch.clamp(visibility * 10, min=0, max=1) * (1 - brightness)).item()
            color = (brightness, 0, 0, alpha)
            triangle = plt.Polygon(self.triangles[i, :, 0:2].clone().detach().cpu(), color=color)
            plt.gca().add_patch(triangle)
        plt.xlim(-2, 2)
        plt.ylim(-1, 1)

        if (target is not None):
            target_cpu = target.clone().detach().cpu()
            for line in range(target_cpu.size(dim=0)):
                print(
                    target_cpu[line,0,0].item(),
                    target_cpu[line,0,1].item(),
                    target_cpu[line,1,0].item(),
                    target_cpu[line,1,1].item())
                plt.plot(
                    (target_cpu[line,0,0].item(), target_cpu[line,1,0].item()),
                    (target_cpu[line,0,1].item(), target_cpu[line,1,1].item()),
                    'b'
                )
        plt.show()

    def render(self, img):
        # TODO
        w = img.size(dim=1)
        h = img.size(dim=2)
        grid_x, grid_y = torch.meshgrid(
                torch.linspace(0, 1, width),
                torch.linspace(0, 1, height))
        coords = torch.dstack((grid_x, grid_y)) # [W x H x 2]
        coords_list = torch.reshape(coords, (w*h, 2)) # [W*H x 2]

        # Calculate the area of the parallelogram formed by the triangle
        area = signed_dist_to_edge(self.triangles[:, 2, :], self.triangles[:, 1, :], self.triangles[:, None, 0, :])

        # Evaluate the edge functions at every position.
        # We should get a [N x P] vector out of each.
        w0 = -signed_dist_to_edge(d[:, 1, :], d[:, 2, :], grid) / area
        w1 = -signed_dist_to_edge(d[:, 2, :], d[:, 0, :], grid) / area
        w2 = -signed_dist_to_edge(d[:, 0, :], d[:, 1, :], grid) / area

        # Only pixels inside the triangles will have color
        mask = (w0 > 0) & (w1 > 0) & (w2 > 0) # [N x P]



suzanne = Mesh([
    load_obj("assets/suzanne0.obj"),
    load_obj("assets/suzanne1.obj"),
    load_obj("assets/suzanne2.obj")
])
target = load_svg("assets/guides.svg", out_width=4, out_height=2)

optimizer = torch.optim.Adam([suzanne.blend_weights], lr=1e-2)
# Run 100 Adam iterations.
for t in range(100):
    print('iteration:', t)
    optimizer.zero_grad()
    suzanne.update_triangles()
    loss = torch.pow(abs(suzanne.triangles.sum()) + 1, -1)

    loss.backward()

    # Take a gradient descent step.
    optimizer.step()
    with torch.no_grad():
        suzanne.constrain()
    print(suzanne.blend_weights)
    if t % 5 == 0:
        suzanne.display(target)

suzanne.output('blended.obj')

# img = empty_image(256, 256)
