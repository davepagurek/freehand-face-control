import torch
import torchvision

device = torch.device("cpu") # TODO: change to "mps" for macbook

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

def empty_image(width, height):
    return torch.zeros([3, width, height])

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
        self.blend_shapes = [(v[0] - rest_vertices).detach() for v in models[1:]]
        self.blend_weights = torch.tensor([0.0 for v in self.blend_shapes], requires_grad=True, device=device)
        self.triangles = torch.zeros([len(self.faces), 3, 3]).to(device)
        self.update_triangles()

    def update_triangles(self):
        self.vertices = self.rest_vertices.clone().detach()
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
    print(suzanne.blend_weights)

suzanne.output('blended.obj')

# img = empty_image(256, 256)